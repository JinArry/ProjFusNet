import os
import torch
import esm
import pandas as pd
import numpy as np
import biotite.structure.io as bsio
from tqdm import tqdm
import logging
import time
from pathlib import Path
import re
import argparse
import sys
import gc
import warnings
from concurrent.futures import ThreadPoolExecutor
import psutil
import json
from Bio import SeqIO

# 设置PyTorch内存分配策略以避免内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 过滤掉特定的FutureWarning
warnings.filterwarnings("ignore", message=".*torch\.cuda\.amp\.autocast.*is deprecated.*", category=FutureWarning)

# Default logging setup - will be overridden in main()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_system_info():
    """获取系统资源信息"""
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
        "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
        "memory_percent": psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
            "gpu_memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
            "gpu_memory_cached": torch.cuda.memory_reserved(0) / (1024**3)  # GB
        })
    
    return info

def log_system_info():
    """记录系统资源信息"""
    info = get_system_info()
    logging.info("System Information:")
    for key, value in info.items():
        logging.info(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

def setup_directories():
    """Create necessary directories for data storage."""
    dirs = ["./Data", "./Data/structure_local", "./Data/results", "./Data/logs"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory ensured: {dir_path}")

def validate_sequence(sequence):
    """
    Validate that a protein sequence contains only valid amino acid codes.
    Returns cleaned sequence and whether it was valid.
    """
    # Standard amino acids in single letter code + X for unknown
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    
    # Remove any whitespace and convert to uppercase
    clean_seq = re.sub(r'\s+', '', sequence).upper()
    
    # Check if all characters are valid amino acids
    is_valid = all(aa in valid_aa for aa in clean_seq)
    
    return clean_seq, is_valid

def manual_extract_b_factors(pdb_file):
    """手动从 PDB 文件中提取 b-factor 值"""
    with open(pdb_file, 'r') as f:
        pdb_lines = f.readlines()
    
    b_factors = []
    for line in pdb_lines:
        if line.startswith('ATOM'):
            try:
                # PDB格式中b-factor通常在第61-66列
                b_factor = float(line[60:66].strip())
                b_factors.append(b_factor)
            except:
                pass
    
    if not b_factors:
        return None
    
    return float(np.mean(b_factors))

def validate_pdb_file(pdb_file):
    """验证PDB文件的有效性"""
    try:
        # 检查文件是否存在
        if not os.path.exists(pdb_file):
            return False, "File does not exist"
        
        # 检查文件大小
        if os.path.getsize(pdb_file) == 0:
            return False, "File is empty"
        
        # 尝试加载结构
        try:
            struct = bsio.load_structure(str(pdb_file))
            if len(struct) == 0:
                return False, "No atoms in structure"
        except Exception as e:
            return False, f"Failed to load structure: {str(e)}"
        
        # 检查是否包含ATOM记录
        with open(pdb_file, 'r') as f:
            content = f.read()
            if 'ATOM' not in content:
                return False, "No ATOM records found"
        
        return True, "Valid PDB file"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def predict_structure_batch(sequences, sequence_ids=None, sequence_labels=None, batch_size=5, output_dir="./Data/structure_local", 
                            save_metrics=True, chunk_size=None, use_mixed_precision=True, default_label="unknown", max_retries=5, retry_delay=10,
                            memory_efficient=False):
    """
    Predict protein structures for a batch of sequences using ESM-Fold.
    
    Args:
        sequences: List of protein sequences
        batch_size: Number of sequences to process at once
        output_dir: Directory to save PDB files
        save_metrics: Whether to save prediction metrics to CSV
        chunk_size: Optional chunk size for model's axial attention
        max_retries: Maximum number of retries for failed predictions
        retry_delay: Delay in seconds between retries
    
    Returns:
        DataFrame with sequences and their pLDDT scores
    """
    # 记录系统信息
    log_system_info()
    
    # 检查GPU可用性并选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 打印设备信息
    if "cuda" in device:
        gpu_id = 0  # 默认使用第一个GPU
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9  # GB
        logging.info(f"使用GPU: {gpu_name} 内存 {gpu_memory:.2f} GB")
        
        # 彻底清理显存
        torch.cuda.empty_cache()
        gc.collect()  # 强制Python垃圾回收
        logging.info("已清理GPU缓存以准备结构预测")
        
        # 记录可用内存
        free_memory = torch.cuda.mem_get_info()[0] / 1e9  # GB
        logging.info(f"当前可用GPU内存: {free_memory:.2f} GB")
    else:
        logging.warning("CUDA不可用，使用CPU进行预测（这将非常慢）")
    
    # 初始化模型
    try:
        # 使用内存优化模式时更加激进地处理内存
        if memory_efficient and torch.cuda.is_available():
            logging.info("使用内存优化模式")
            # 使用较小的默认chunk_size
            default_chunk_size = 32 if chunk_size is None else chunk_size
            # 启用ESM-Fold的内置内存优化
            model = esm.pretrained.esmfold_v1()
            model = model.eval()
            model = model.cuda()
            
            # 使用torch.compile优化内存使用（如果可用）
            if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
                logging.info("使用torch.compile优化模型性能")
                model = torch.compile(model, mode="reduce-overhead")
                
            # 启用强制梯度检查点
            if hasattr(model, "trunk") and hasattr(model.trunk, "enable_checkpointing"):
                model.trunk.enable_checkpointing()
                logging.info("已启用梯度检查点以优化内存使用")
            
            # 设置默认chunk_size
            if hasattr(model, "set_chunk_size"):
                model.set_chunk_size(default_chunk_size)
                logging.info(f"全局设置chunk_size={default_chunk_size} (内存优化模式)")
        else:
            # 标准模式
            model = esm.pretrained.esmfold_v1()
            model = model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                logging.info("使用CUDA进行预测")
                
                # 启用梯度检查点以优化内存使用
                if hasattr(model, "trunk") and hasattr(model.trunk, "enable_checkpointing"):
                    model.trunk.enable_checkpointing()
                    logging.info("已启用梯度检查点以优化内存使用")
                    
                # 设置默认chunk_size（如果没有指定）
                if chunk_size is not None and hasattr(model, "set_chunk_size"):
                    model.set_chunk_size(chunk_size)
                    logging.info(f"全局设置chunk_size={chunk_size}")
            else:
                logging.info("CUDA不可用，使用CPU（这将很慢）")
    except Exception as e:
        logging.error(f"Error initializing ESM-Fold model: {e}")
        raise
    
    results = []
    
    # 创建批处理模式，使用真正的批处理来提高效率
    # 预处理所有序列
    processed_sequences = []
    processed_ids = []
    processed_labels = []
    pdb_files = []
    skipped_indices = []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 第一步：预处理序列并检查已存在的结果
    logging.info("预处理序列并检查现有预测结果")
    with tqdm(total=len(sequences), desc="Preprocessing sequences") as pbar:
        for idx, sequence in enumerate(sequences):
            sequence_id = sequence_ids[idx] if sequence_ids else f"seq_{idx+1}"
            sequence_label = sequence_labels[idx] if sequence_labels and idx < len(sequence_labels) else default_label
            
            # 清理和验证序列
            clean_seq, is_valid = validate_sequence(sequence)
            
            if not is_valid:
                logging.warning(f"序列 {sequence_id} 包含无效的氨基酸代码。将使用清理后的序列继续。")
            
            if not clean_seq:
                logging.error(f"序列 {sequence_id} 在清理后为空。跳过。")
                results.append({
                    "sequence": sequence,
                    "sequence_id": sequence_id,
                    "pdb_file": None,
                    "pLDDT": np.nan,
                    "length": len(sequence),
                    "label": sequence_label,
                    "retries": 0,  # 没有重试
                    "status": "error",
                    "error": "清理后序列为空"
                })
                pbar.update(1)
                continue
            
            # 创建输出文件名
            seq_hash = str(abs(hash(clean_seq)) % 10000000)
            pdb_file = Path(output_dir) / f"{sequence_id}_{seq_hash}.pdb"
            
            # 检查文件是否存在并有效
            existing_pdb = pdb_file
            if existing_pdb and os.path.exists(existing_pdb):
                try:
                    # 验证PDB文件
                    is_valid, validation_msg = validate_pdb_file(existing_pdb)
                    if is_valid:
                        # 验证PDB文件并读取pLDDT值 - 处理biotite API变化
                        try:
                            struct = bsio.load_structure(str(pdb_file))
                            # 尝试多种可能的属性名
                            if hasattr(struct, 'b_factors'):
                                b_factors = struct.b_factors
                                plddt = float(np.mean(b_factors))
                            elif hasattr(struct, 'b_factor'):  
                                b_factors = struct.b_factor
                                plddt = float(np.mean(b_factors))
                            else:
                                # 如果没有找到b_factor属性，尝试加载时明确指定
                                struct = bsio.load_structure(str(pdb_file), extra_fields=["b_factor"])
                                if hasattr(struct, 'b_factor'):
                                    b_factors = struct.b_factor
                                    plddt = float(np.mean(b_factors))
                                else:
                                    # 仍然找不到，则尝试手动提取
                                    plddt = manual_extract_b_factors(pdb_file)
                                    if plddt is None:
                                        raise ValueError("无法从PDB文件提取b-factor值")
                        except Exception as e:
                            logging.error(f"读取PDB文件{pdb_file}的b-factor时出错: {e}")
                            raise
                        
                        # 添加到已处理列表
                        logging.info(f"加载现有预测结果：{sequence_id}: pLDDT = {plddt:.2f}")
                        
                        results.append({
                            "sequence": sequence,
                            "sequence_id": sequence_id,
                            "pdb_file": str(pdb_file),
                            "pLDDT": plddt,
                            "length": len(sequence),
                            "label": sequence_label,
                            "retries": 0,  # 没有重试
                            "status": "existing"
                        })
                        pbar.update(1)
                        continue  # 跳过已处理的序列
                    else:
                        logging.warning(f"现有PDB文件无效 ({validation_msg})，将重新计算")
                except Exception as e:
                    logging.warning(f"加载现有预测结果时出错：{e}。将重新计算。")
            
            # 记录需要处理的序列
            processed_sequences.append(clean_seq)
            processed_ids.append(sequence_id)
            processed_labels.append(sequence_label)
            pdb_files.append(pdb_file)
            pbar.update(1)
    
    # 如果没有需要处理的序列，直接返回结果
    if not processed_sequences:
        logging.info("所有序列已有现有预测结果，不需要进一步处理")
        return pd.DataFrame(results)
    
    # 按批次处理序列
    logging.info(f"使用批处理模式处理{len(processed_sequences)}个序列，批大小为{batch_size}")
    with tqdm(total=len(processed_sequences), desc="Predicting structures") as pbar:
        for i in range(0, len(processed_sequences), batch_size):
            batch_seqs = processed_sequences[i:i+batch_size]
            batch_ids = processed_ids[i:i+batch_size]
            batch_labels = processed_labels[i:i+batch_size]
            batch_pdb_files = pdb_files[i:i+batch_size]
            
            logging.info(f"处理批次 {i//batch_size + 1}/{(len(processed_sequences)-1)//batch_size + 1}，共{len(batch_seqs)}个序列")
            
            # 为每个序列单独设置chunk_size并预测结构
            for seq_idx, (seq, seq_id, seq_label, pdb_file) in enumerate(zip(batch_seqs, batch_ids, batch_labels, batch_pdb_files)):
                # 创建PDB文件输出路径
                seq_hash = str(abs(hash(seq)) % 10000000)
                output_path = Path(output_dir) / f"{seq_id}_{seq_hash}.pdb"
                # 找到原始序列信息
                orig_idx = sequences.index(seq) if seq in sequences else -1
                orig_seq = sequences[orig_idx] if orig_idx >= 0 else seq
                
                # 根据序列长度动态设置chunk_size
                seq_length = len(seq)
                adaptive_chunk_size = None
                
                # 在内存优化模式下使用更小的chunk_size
                if chunk_size is None:  # 只有在用户没有指定chunk_size时才自动设置
                    if memory_efficient:
                        # 内存优化模式下使用更小的chunk_size
                        if seq_length > 1000:
                            adaptive_chunk_size = 64
                        elif seq_length > 500:
                            adaptive_chunk_size = 32
                        elif seq_length > 200:
                            adaptive_chunk_size = 16
                        else:
                            adaptive_chunk_size = 8
                    else:
                        # 标准模式
                        if seq_length > 1000:
                            adaptive_chunk_size = 128
                        elif seq_length > 500:
                            adaptive_chunk_size = 64
                        elif seq_length > 200:
                            adaptive_chunk_size = 32
                elif seq_length > 250:
                    adaptive_chunk_size = 32
                else:
                    adaptive_chunk_size = 16
                    if hasattr(model, "set_chunk_size"):
                        model.set_chunk_size(adaptive_chunk_size)
                        logging.info(f"为序列 {seq_id} (长度={seq_length}) 设置chunk_size={adaptive_chunk_size}")
                
                # 初始化状态并尝试预测（不包含重试）
                try:
                    # 开始计时
                    start_time = time.time()
                    
                    # 使用混合精度和无梯度模式进行推理
                    # 确保特征矩阵使用float32类型，而不是float64
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_mixed_precision and torch.cuda.is_available()):
                        # 确保输入数据类型符合AlphaFold要求
                        # 1. 氨基酸编码需严格遵循21维(0-20)范围
                        # 2. 特征矩阵必须包含正确的批次维度
                        # 3. 浮点数特征必须使用float32类型，不能使用float64
                        # 4. 需要移除对象类型(np.dtype('O'))的特征
                        out = model.infer_pdb(seq)
                    
                    # 计算总用时
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # 保存PDB文件
                    with open(output_path, "w") as f:
                        f.write(out)
                    
                    # 验证生成的PDB文件
                    is_valid, validation_msg = validate_pdb_file(output_path)
                    if not is_valid:
                        raise ValueError(f"生成的PDB文件无效: {validation_msg}")
                    
                    # 计算pLDDT平均值 - 有多种方法尝试提取b-factor
                    avg_plddt = None
                    extraction_methods = [
                        # 方法1: 直接使用b_factors属性
                        lambda: (float(np.mean(bsio.load_structure(output_path).b_factors)) 
                                 if hasattr(bsio.load_structure(output_path), 'b_factors') else None),
                        # 方法2: 直接使用b_factor属性
                        lambda: (float(np.mean(bsio.load_structure(output_path).b_factor)) 
                                 if hasattr(bsio.load_structure(output_path), 'b_factor') else None),
                        # 方法3: 指定额外字段加载
                        lambda: (float(np.mean(bsio.load_structure(output_path, extra_fields=["b_factor"]).b_factor)) 
                                 if hasattr(bsio.load_structure(output_path, extra_fields=["b_factor"]), 'b_factor') else None),
                        # 方法4: 手动解析PDB文件
                        lambda: manual_extract_b_factors(output_path)
                    ]
                    
                    # 尝试每种方法
                    for method_index, extract_method in enumerate(extraction_methods):
                        try:
                            extracted_plddt = extract_method()
                            if extracted_plddt is not None:
                                avg_plddt = extracted_plddt
                                logging.info(f"使用方法{method_index+1}成功提取pLDDT分数")
                                break
                        except Exception as e:
                            logging.debug(f"方法{method_index+1}提取pLDDT失败: {e}")
                            continue
                    
                    # 如果所有方法都失败，抛出异常
                    if avg_plddt is None:
                        raise ValueError("所有提取pLDDT分数的方法都失败了")
                    
                    # 记录成功信息
                    logging.info(f"预测 {seq_id}: pLDDT = {avg_plddt:.2f}, 耗时 = {elapsed_time:.1f}s")
                    
                    # 添加成功结果
                    results.append({
                        "sequence": orig_seq,
                        "sequence_id": seq_id,
                        "pdb_file": str(output_path),
                        "pLDDT": avg_plddt,
                        "length": len(seq),
                        "prediction_time": elapsed_time,
                        "label": seq_label,
                        "retries": 0,
                        "status": "success"
                    })
                    
                    # 每个序列预测后清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # 内存优化模式下进行更彻底的清理
                        if memory_efficient:
                            gc.collect()
                            # 显示当前可用内存
                            free_memory = torch.cuda.mem_get_info()[0] / 1e9  # GB
                            logging.info(f"预测后可用GPU内存: {free_memory:.2f} GB")
                    
                except Exception as e:
                    # 记录失败信息，准备后续重试
                    logging.warning(f"预测序列 {seq_id} 首次尝试失败: {e}")
                    
                    # 添加到失败列表，准备后续重试
                    results.append({
                        "sequence": orig_seq,
                        "sequence_id": seq_id,
                        "pdb_file": str(output_path),
                        "pLDDT": np.nan,
                        "length": len(seq),
                        "label": seq_label,
                        "retries": 0,  # 还没有重试
                        "status": "retry",  # 标记为需要重试
                        "error": str(e),
                        "adaptive_chunk_size": adaptive_chunk_size,
                        "seq": seq  # 保存序列以便重试
                    })
                
                pbar.update(1)
    
    # 处理需要重试的序列
    # 首先，找出首轮失败需要重试的序列
    retry_results = [r for r in results if r["status"] == "retry"]
    logging.info(f"首轮预测完成，{len(retry_results)}/{len(results)}个序列需要重试")
    
    # 固定进行5轮重试
    for retry_round in range(1, 6):
        if not retry_results:
            logging.info(f"第{retry_round}轮重试：没有需要重试的序列，但仍将继续后续轮次")
        else:
            logging.info(f"开始第{retry_round}轮重试，共{len(retry_results)}个序列")
        
        # 清理GPU内存以准备重试
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            free_memory = torch.cuda.mem_get_info()[0] / 1e9  # GB
            logging.info(f"重试前可用GPU内存: {free_memory:.2f} GB")
        
        # 等待一段时间后开始重试
        if retry_delay > 0:
            logging.info(f"等待 {retry_delay} 秒后开始第{retry_round}轮重试...")
            time.sleep(retry_delay)
        
        # 记录此轮要处理的序列
        current_retry_results = retry_results.copy()
        retry_results = []  # 清空列表，用于收集本轮中失败的序列（只收集当前轮失败的，而不是所有之前失败的）
        
        # 按批次重试
        with tqdm(total=len(current_retry_results), desc=f"Retry round {retry_round}") as pbar:
            for i in range(0, len(current_retry_results), batch_size):
                batch_retry = current_retry_results[i:i+batch_size]
                logging.info(f"重试批次 {i//batch_size + 1}/{(len(current_retry_results)-1)//batch_size + 1}，共{len(batch_retry)}个序列")
                
                # 处理批次中的每个序列
                for retry_item in batch_retry:
                    seq_id = retry_item["sequence_id"]
                    seq = retry_item["seq"]
                    orig_seq = retry_item["sequence"]
                    seq_label = retry_item["label"]
                    output_path = retry_item["pdb_file"]
                    adaptive_chunk_size = retry_item["adaptive_chunk_size"]
                    
                    logging.info(f"重试 {retry_round}/{max_retries} 预测序列 {seq_id}")
                    
                    # 尝试用不同的chunk_size重试
                    if adaptive_chunk_size is not None and adaptive_chunk_size > 16:
                        # 减小chunk_size以减少内存需求
                        new_chunk_size = max(16, adaptive_chunk_size // 2)
                        if hasattr(model, "set_chunk_size"):
                            model.set_chunk_size(new_chunk_size)
                            logging.info(f"重试时为序列 {seq_id} 减小chunk_size: {adaptive_chunk_size} -> {new_chunk_size}")
                            adaptive_chunk_size = new_chunk_size
                    
                    try:
                        # 开始计时
                        start_time = time.time()
                        
                        # 使用混合精度和无梯度模式进行推理
                        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_mixed_precision and torch.cuda.is_available()):
                            out = model.infer_pdb(seq)
                        
                        # 计算总用时
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                        # 保存PDB文件
                        with open(output_path, "w") as f:
                            f.write(out)
                        
                        # 验证生成的PDB文件
                        is_valid, validation_msg = validate_pdb_file(output_path)
                        if not is_valid:
                            raise ValueError(f"生成的PDB文件无效: {validation_msg}")
                        
                        # 计算pLDDT平均值
                        avg_plddt = None
                        extraction_methods = [
                            lambda: (float(np.mean(bsio.load_structure(output_path).b_factors)) 
                                     if hasattr(bsio.load_structure(output_path), 'b_factors') else None),
                            lambda: (float(np.mean(bsio.load_structure(output_path).b_factor)) 
                                     if hasattr(bsio.load_structure(output_path), 'b_factor') else None),
                            lambda: (float(np.mean(bsio.load_structure(output_path, extra_fields=["b_factor"]).b_factor)) 
                                     if hasattr(bsio.load_structure(output_path, extra_fields=["b_factor"]), 'b_factor') else None),
                            lambda: manual_extract_b_factors(output_path)
                        ]
                        
                        for method_index, extract_method in enumerate(extraction_methods):
                            try:
                                extracted_plddt = extract_method()
                                if extracted_plddt is not None:
                                    avg_plddt = extracted_plddt
                                    logging.info(f"使用方法{method_index+1}成功提取pLDDT分数")
                                    break
                            except Exception as e:
                                logging.debug(f"方法{method_index+1}提取pLDDT失败: {e}")
                                continue
                        
                        if avg_plddt is None:
                            raise ValueError("所有提取pLDDT分数的方法都失败了")
                        
                        # 记录成功信息
                        logging.info(f"预测 {seq_id}: pLDDT = {avg_plddt:.2f}, 耗时 = {elapsed_time:.1f}s, 重试 {retry_round} 次后成功")
                        
                        # 更新结果（替换原始失败记录）
                        for i, r in enumerate(results):
                            if r["sequence_id"] == seq_id and r["status"] == "retry":
                                results[i] = {
                                    "sequence": orig_seq,
                                    "sequence_id": seq_id,
                                    "pdb_file": str(output_path),
                                    "pLDDT": avg_plddt,
                                    "length": len(seq),
                                    "prediction_time": elapsed_time,
                                    "label": seq_label,
                                    "retries": retry_round,
                                    "status": "success"
                                }
                                break
                        
                        # 每个序列预测后清理GPU内存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if memory_efficient:
                                gc.collect()
                        
                    except Exception as e:
                        # 记录失败信息
                        logging.warning(f"预测序列 {seq_id} 重试 {retry_round}/{max_retries} 失败: {e}")
                        
                        # 添加到下一轮重试列表
                        if retry_round < 5:  # 固定5轮重试
                            retry_item["retries"] = retry_round
                            retry_item["adaptive_chunk_size"] = adaptive_chunk_size
                            retry_item["error"] = str(e)
                            retry_results.append(retry_item)  # 只有本轮失败的会被添加到下一轮
                        else:
                            # 已完成所有重试轮次，标记为最终失败
                            for i, r in enumerate(results):
                                if r["sequence_id"] == seq_id and r["status"] == "retry":
                                    results[i] = {
                                        "sequence": orig_seq,
                                        "sequence_id": seq_id,
                                        "pdb_file": str(output_path),
                                        "pLDDT": np.nan,
                                        "length": len(seq),
                                        "label": seq_label,
                                        "retries": max_retries,
                                        "status": "error",
                                        "error": str(e)
                                    }
                                    break
                            logging.error(f"预测序列 {seq_id} 的结构失败，已重试 {max_retries} 次: {e}")
                    
                    pbar.update(1)
    
    # 将剩余的需要重试的序列标记为失败
    for retry_item in retry_results:
        seq_id = retry_item["sequence_id"]
        for i, r in enumerate(results):
            if r["sequence_id"] == seq_id and r["status"] == "retry":
                results[i]["status"] = "error"
                results[i]["retries"] = 5  # 固定为5轮重试
    
    # 清理结果中的临时字段
    for r in results:
        if "seq" in r:
            del r["seq"]
        if "adaptive_chunk_size" in r:
            del r["adaptive_chunk_size"]
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save metrics if requested
    if save_metrics and not results_df.empty:
        metrics_file = Path("./Data/results/prediction_metrics.csv")
        results_df.to_csv(metrics_file, index=False)
        logging.info(f"Saved prediction metrics to {metrics_file}")
        
        # 保存详细的统计信息
        stats = {
            "total_sequences": len(results_df),
            "successful_predictions": (results_df['status'] == 'success').sum(),
            "failed_predictions": (results_df['status'] == 'error').sum(),
            "existing_predictions": (results_df['status'] == 'existing').sum(),
            "average_plddt": results_df['pLDDT'].mean(),
            "median_plddt": results_df['pLDDT'].median(),
            "min_plddt": results_df['pLDDT'].min(),
            "max_plddt": results_df['pLDDT'].max(),
            "total_retries": results_df['retries'].sum(),
            "successful_with_retry": ((results_df['status'] == 'success') & (results_df['retries'] > 0)).sum(),
            "system_info": get_system_info()
        }
        
        stats_file = Path("./Data/results/prediction_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Saved detailed statistics to {stats_file}")
    
    return results_df

def setup_logging(log_file=None):
    """Set up logging with optional file output"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def read_fasta_file(fasta_file):
    """
    从FASTA文件中读取序列数据
    
    Args:
        fasta_file: FASTA文件路径
        
    Returns:
        tuple: (sequences, sequence_ids)
    """
    sequences = []
    sequence_ids = []
    
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            # 只提取序列ID和序列内容
            seq_id = record.id.split("|")[0]  # 只保留ID的第一部分，去掉标签信息
            sequences.append(str(record.seq))
            sequence_ids.append(seq_id)
            
        logging.info(f"从FASTA文件读取了 {len(sequences)} 个序列")
        return sequences, sequence_ids
    except Exception as e:
        logging.error(f"读取FASTA文件时出错: {e}")
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Predict protein structures using ESM-Fold")
    
    # Input options - either sequence, file, or fasta
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-s", "--sequence", type=str, help="A single protein sequence")
    input_group.add_argument("-f", "--file", type=str, 
                          help="Path to CSV file containing protein sequences (must have 'sequence' column)")
    input_group.add_argument("--fasta", type=str,
                          help="Path to FASTA file containing protein sequences")
    
    # Processing options
    parser.add_argument("-o", "--output-dir", type=str, default="./Data/structure_local",
                      help="Directory to save PDB files (default: ./Data/structure_local)")
    parser.add_argument("-b", "--batch-size", type=int, default=5,
                      help="Number of sequences to process in one batch (default: 5)")
    parser.add_argument("-c", "--chunk-size", type=int, default=128,
                      help="Chunk size for ESM-Fold's axial attention (default: 128)")
    
    # Other options
    parser.add_argument("--log-file", type=str, default="protein_prediction.log",
                      help="Path to log file (default: protein_prediction.log)")
    parser.add_argument("--no-save-metrics", action="store_true",
                      help="Don't save prediction metrics to CSV file")
    parser.add_argument("--no-mixed-precision", action="store_true",
                      help="Disable mixed precision inference (reduces speed but may increase accuracy)")
    parser.add_argument("--label", type=str, default="unknown",
                      help="Default label to assign to all sequences in the output CSV (default: unknown)")
    parser.add_argument("--max-retries", type=int, default=5,
                      help="Maximum number of retries for failed predictions (default: 5)")
    parser.add_argument("--retry-delay", type=int, default=10,
                      help="Delay in seconds between retries (default: 10)")
    parser.add_argument("--memory-efficient", action="store_true",
                      help="Enable additional memory optimization techniques (may be slower)")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_file)
    
    # Create necessary directories
    setup_directories()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        seq_list = []
        id_list = None
        
        # Get sequences from either command line, CSV file, or FASTA file
        if args.sequence:
            # Single sequence from command line
            seq_list = [args.sequence]
            logging.info("Using single sequence from command line")
        elif args.fasta:
            # Sequences from FASTA file
            if not os.path.exists(args.fasta):
                logging.error(f"FASTA file not found: {args.fasta}")
                print(f"Error: FASTA file not found: {args.fasta}")
                return 1
            
            try:
                seq_list, id_list = read_fasta_file(args.fasta)
                logging.info(f"Loaded {len(seq_list)} sequences from {args.fasta}")
            except Exception as e:
                logging.error(f"Error loading FASTA file: {e}")
                print(f"Error loading FASTA file: {e}")
                return 1
        else:
            # Sequences from CSV file
            if not os.path.exists(args.file):
                logging.error(f"Data file not found: {args.file}")
                print(f"Error: Data file not found: {args.file}")
                return 1
                
            try:
                data = pd.read_csv(args.file)
                if 'sequence' not in data.columns:
                    logging.error("No 'sequence' column found in data file")
                    print("Error: No 'sequence' column found in data file")
                    return 1
                    
                seq_list = data['sequence'].tolist()
                
                # 检查是否存在ID列（优先使用uniProtkbId列）
                if 'uniProtkbId' in data.columns:
                    id_list = data['uniProtkbId'].tolist()
                    logging.info(f"Found uniProtkbId column in {args.file}, will use these IDs for PDB file naming")
                elif 'ID' in data.columns:
                    id_list = data['ID'].tolist()
                    logging.info(f"Found ID column in {args.file}, will use IDs for PDB file naming")
                
                logging.info(f"Loaded {len(seq_list)} sequences from {args.file}")
            except Exception as e:
                logging.error(f"Error loading CSV file: {e}")
                print(f"Error loading CSV file: {e}")
                return 1
        
        # Run prediction
        results = predict_structure_batch(
            sequences=seq_list, 
            sequence_ids=id_list,
            sequence_labels=None,  # 不再使用标签信息
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir,
            save_metrics=not args.no_save_metrics,
            use_mixed_precision=not args.no_mixed_precision,
            default_label=args.label,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            memory_efficient=args.memory_efficient
        )
        
        # Print summary
        success_count = (results['status'] == 'success').sum()
        error_count = (results['status'] == 'error').sum()
        mean_plddt = results['pLDDT'].mean()
        
        # 计算重试统计
        retry_counts = results['retries'].fillna(0).astype(int)
        total_retries = retry_counts.sum()
        successful_with_retry = ((results['status'] == 'success') & (retry_counts > 0)).sum()
        
        print(f"\nResults Summary:")
        print(f"Total sequences: {len(results)}")
        print(f"Successful predictions: {success_count}")
        print(f"Failed predictions: {error_count}")
        print(f"Average pLDDT score: {mean_plddt:.2f}")
        print(f"Total retry attempts: {total_retries}")
        print(f"Predictions successful after retry: {successful_with_retry}")
        
        return 0
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())