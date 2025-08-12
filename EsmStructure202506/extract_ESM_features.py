#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取蛋白质序列的ESM-2特征

该脚本从训练集和测试集CSV文件中读取蛋白质序列，
使用ESM-2预训练模型提取嵌入特征，
然后将这些特征保存到新的文件中，以供后续模型使用。

使用方法:
python extract_esm_features.py --train_file train_features.csv --test_file test_features.csv
"""

import os
import sys
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
import time
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("esm_feature_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 检查ESM-2模块是否可用
try:
    import esm

    ESM_AVAILABLE = True
    logger.info("成功导入ESM-2模块")
except ImportError:
    ESM_AVAILABLE = False
    logger.error("未能导入ESM-2模块，请使用pip install 'fair-esm'安装")
    sys.exit(1)


def log_important(message):
    """打印重要日志信息，使其更加醒目"""
    separator = "=" * 50
    logger.info(separator)
    logger.info(message)
    logger.info(separator)
    # 同时输出到控制台，以确保可见性
    print(f"\n{separator}\n{message}\n{separator}\n")


def extract_esm_features(sequences, ids, model_name="esm2_t33_650M_UR50D", batch_size=4,
                         device=None, gpu=None, repr_layer=33):
    """
    使用ESM-2模型从蛋白质序列中提取特征

    Args:
        sequences: 蛋白质序列列表
        ids: 序列ID列表
        model_name: ESM-2模型名称
        batch_size: 批处理大小
        device: 计算设备
        gpu: GPU设备ID
        repr_layer: 提取特征的表示层索引

    Returns:
        特征字典，键为序列ID，值为特征向量
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif gpu is not None:
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    logger.info(f"使用设备: {device}")

    # 加载ESM-2模型和分词器
    log_important(f"加载ESM-2模型: {model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()  # 设置为评估模式
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    # 跟踪进度并准备结果字典
    results = {}
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), total=total_batches, desc="提取ESM特征"):
            batch_sequences = sequences[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # 准备批次数据
            batch_data = [(f"protein_{j}", seq) for j, seq in enumerate(batch_sequences)]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            # 提取特征
            try:
                results_batch = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
                embeddings = results_batch["representations"][repr_layer].cpu().numpy()

                # 对于每个序列，使用平均池化得到全局表示
                for j, (seq_id, seq) in enumerate(zip(batch_ids, batch_sequences)):
                    # 取有效token的表示（排除填充和特殊token）
                    seq_len = len(seq)
                    # ESM-2 token索引: 0=<cls>, 1...seq_len=序列, seq_len+1=<eos>
                    seq_embedding = embeddings[j, 1:seq_len + 1]  # 排除<cls>和<eos>
                    # 计算平均值作为全局表示
                    global_embedding = np.mean(seq_embedding, axis=0)
                    results[seq_id] = global_embedding
            except Exception as e:
                logger.error(f"处理批次时出错: {e}")
                # 如果是内存错误，尝试减小批次大小
                if isinstance(e, torch.cuda.OutOfMemoryError) and batch_size > 1:
                    logger.warning(f"CUDA内存不足。减小批次大小并重试...")
                    # 重置内存
                    torch.cuda.empty_cache()
                    # 处理单个序列
                    for j, (seq, seq_id) in enumerate(zip(batch_sequences, batch_ids)):
                        try:
                            single_batch = [(f"protein_{j}", seq)]
                            _, _, tokens = batch_converter(single_batch)
                            tokens = tokens.to(device)

                            result = model(tokens, repr_layers=[repr_layer], return_contacts=False)
                            embedding = result["representations"][repr_layer].cpu().numpy()

                            # 取有效token的表示
                            seq_len = len(seq)
                            seq_embedding = embedding[0, 1:seq_len + 1]  # 排除<cls>和<eos>
                            global_embedding = np.mean(seq_embedding, axis=0)
                            results[seq_id] = global_embedding

                            logger.info(f"成功处理序列 {seq_id}")
                        except Exception as inner_e:
                            logger.error(f"处理序列 {seq_id} 时出错: {inner_e}")

            # 定期清理GPU内存
            if torch.cuda.is_available() and (i + batch_size) % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

    logger.info(f"成功提取 {len(results)}/{len(sequences)} 个序列的ESM特征")
    return results


def process_data_file(file_path, output_file, model_name="esm2_t33_650M_UR50D", batch_size=4, device=None):
    """
    处理数据文件并提取ESM特征

    Args:
        file_path: 数据文件路径
        output_file: 输出文件路径
        model_name: ESM-2模型名称
        batch_size: 批处理大小
        device: 计算设备
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False

    try:
        # 读取数据
        df = pd.read_csv(file_path)
        logger.info(f"从 {file_path} 加载了 {len(df)} 条记录")

        # 检查必要的列
        if 'peptide_id' not in df.columns:
            logger.error(f"文件中缺少peptide_id列: {file_path}")
            return False

        if 'sequence' not in df.columns:
            logger.error(f"文件中缺少sequence列: {file_path}")
            return False

        # 提取序列和ID
        sequences = df['sequence'].tolist()
        ids = df['peptide_id'].tolist()

        # 检查序列是否有效
        valid_indices = []
        for i, seq in enumerate(sequences):
            if isinstance(seq, str) and len(seq) > 0:
                valid_indices.append(i)
            else:
                logger.warning(f"跳过无效序列 (ID: {ids[i]}): {seq}")

        # 过滤无效序列
        if len(valid_indices) < len(sequences):
            logger.warning(f"有 {len(sequences) - len(valid_indices)} 个无效序列被跳过")
            sequences = [sequences[i] for i in valid_indices]
            ids = [ids[i] for i in valid_indices]

        # 提取ESM特征
        log_important(f"开始从 {file_path} 提取ESM特征...")
        esm_features = extract_esm_features(
            sequences=sequences,
            ids=ids,
            model_name=model_name,
            batch_size=batch_size,
            device=device
        )

        # 保存结果
        with open(output_file, 'wb') as f:
            pickle.dump(esm_features, f)
        # 保存为csv格式
        df = pd.DataFrame.from_dict(esm_features, orient='index')
        df.index.name = 'peptide_id'
        csv_file = output_file.replace('.pkl', '.csv')
        df.to_csv(csv_file)
        log_important(f"ESM特征已保存到: {output_file} 和 {csv_file}")

        # 比对pkl和csv内容是否一致
        with open(output_file, 'rb') as f:
            pkl_feats = pickle.load(f)
        df2 = pd.read_csv(csv_file, index_col=0)
        match = True
        for pid, arr in pkl_feats.items():
            arr_csv = df2.loc[pid].values
            if not np.allclose(arr, arr_csv):
                print(f"Mismatch for {pid}")
                match = False
                break
        if match:
            print("pkl和csv特征完全一致！")
        else:
            print("pkl和csv特征有不一致！")

        return True

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def read_fasta_file(fasta_file):
    """
    从FASTA文件中读取序列和ID

    Args:
        fasta_file: FASTA文件路径

    Returns:
        sequences: 序列列表
        ids: 序列ID列表
    """
    sequences = []
    ids = []

    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            # 从序列ID中提取UniProt ID（去掉可能的标签信息）
            seq_id = record.id.split('|')[0] if '|' in record.id else record.id
            sequences.append(str(record.seq))
            ids.append(seq_id)

        logger.info(f"从 {fasta_file} 中读取了 {len(sequences)} 条序列")
        return sequences, ids
    except Exception as e:
        logger.error(f"读取FASTA文件时出错: {e}")
        return [], []


def main():
    """主函数"""
    # 获取当前脚本路径和相关目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置默认的FASTA文件路径
    default_fasta = os.path.join(current_dir, 'Data/combined_dataset.fasta')
    default_output_dir = os.path.join(current_dir, 'Data/esm_features')

    parser = argparse.ArgumentParser(description='使用ESM-2提取蛋白质序列特征')
    parser.add_argument('--fasta', type=str, default=default_fasta,
                        help='FASTA文件路径，包含要提取特征的蛋白质序列')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='输出特征目录')
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D',
                        choices=['esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D',
                                 'esm2_t30_150M_UR50D', 'esm2_t12_35M_UR50D'],
                        help='ESM-2模型名称')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批处理大小')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID')

    args = parser.parse_args()

    # 构建完整路径
    fasta_file = args.fasta
    output_dir = args.output_dir

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置输出文件路径
    output_file = os.path.join(output_dir, 'esm_features.pkl')

    # 配置设备
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 记录基本信息
    log_important("ESM特征提取任务开始")
    logger.info(f"FASTA文件: {fasta_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"ESM模型: {args.model_name}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"计算设备: {device}")

    # 读取FASTA文件
    sequences, ids = read_fasta_file(fasta_file)

    if not sequences:
        logger.error("未能从FASTA文件中读取到任何序列")
        return

    # 提取ESM特征
    log_important("开始提取ESM特征...")
    esm_features = extract_esm_features(
        sequences=sequences,
        ids=ids,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=device
    )

    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(esm_features, f)
    # 保存为csv格式
    df = pd.DataFrame.from_dict(esm_features, orient='index')
    df.index.name = 'peptide_id'
    csv_file = output_file.replace('.pkl', '.csv')
    df.to_csv(csv_file)
    log_important(f"ESM特征已保存到: {output_file} 和 {csv_file}")

    # 比对pkl和csv内容是否一致
    with open(output_file, 'rb') as f:
        pkl_feats = pickle.load(f)
    df2 = pd.read_csv(csv_file, index_col=0)
    match = True
    for pid, arr in pkl_feats.items():
        arr_csv = df2.loc[pid].values
        if not np.allclose(arr, arr_csv):
            print(f"Mismatch for {pid}")
            match = False
            break
    if match:
        print("pkl和csv特征完全一致！")
    else:
        print("pkl和csv特征有不一致！")


if __name__ == "__main__":
    main()
