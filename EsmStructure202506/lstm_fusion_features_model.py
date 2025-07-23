#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质序列LSTM分类器 (集成ESM-2特征的投影融合版本)

该脚本从预处理的特征文件中加载蛋白质特征数据，
并使用ESM-2预训练模型提取额外的特征，
然后通过投影层将不同特征投影至相同维度后融合，
最后使用LSTM神经网络进行分类，
并进行5折交叉验证评估模型性能。

作者：Cascade AI
日期：2025-04-16
"""

import os
import random
import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score, matthews_corrcoef, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Bio import SeqIO
import shutil
import csv
import time

# 设置matplotlib使用支持英文的字体
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"设置字体时出错: {e}, 可能导致显示异常")

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入ESM-2模块
try:
    import esm

    ESM_AVAILABLE = True
    logger.info("成功导入ESM-2模块")
except ImportError:
    ESM_AVAILABLE = False
    logger.warning("未能导入ESM-2模块，请使用pip install 'fair-esm'安装")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# 特征投影层定义
class FeatureProjection(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        """
        特征投影层，将两种不同维度的特征投影到相同的维度空间

        Args:
            input_dim1: 第一种特征的输入维度
            input_dim2: 第二种特征的输入维度
            output_dim: 投影后的输出维度
        """
        super(FeatureProjection, self).__init__()
        self.projection1 = nn.Sequential(
            nn.Linear(input_dim1, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )

        self.projection2 = nn.Sequential(
            nn.Linear(input_dim2, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, feature1, feature2):
        """
        将两种特征投影到相同的维度空间，然后拼接

        Args:
            feature1: 第一种特征 [batch_size, input_dim1]
            feature2: 第二种特征 [batch_size, input_dim2]

        Returns:
            拼接后的特征 [batch_size, output_dim*2]
        """
        projected1 = self.projection1(feature1)
        projected2 = self.projection2(feature2)
        return torch.cat([projected1, projected2], dim=1)


# LSTM模型定义
class LSTMClassifier(nn.Module):
    """
    LSTM分类器
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes, use_embedding=False, vocab_size=21,
                 embedding_dim=64):
        super(LSTMClassifier, self).__init__()

        # 新增氨基酸嵌入层，用于直接处理氨基酸序列
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        else:
            # 原始LSTM层
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # *2是因为双向LSTM
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """前向传播"""
        # 如果使用嵌入层，假设x是氨基酸索引序列 [batch_size, seq_len]
        if self.use_embedding:
            x = self.embedding(x)
        # 兼容输入为[batch, feature_dim]的情况，自动unsqueeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out)
        fc_out = self.relu(fc_out)
        fc_out = self.dropout(fc_out)
        output = self.out(fc_out)
        return output


# 数据集类
class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        """
        蛋白质数据集

        Args:
            features: 特征矩阵 [num_samples, feature_dim]
            labels: 标签向量 [num_samples]
        """
        # 确保数据类型为float32以兼容AlphaFold
        self.features = torch.FloatTensor(features.astype(np.float32))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(0), self.labels[idx]  # 添加seq_len维度 [1, feature_dim]


# 融合特征数据集类
class FusionDataset(Dataset):
    def __init__(self, struct_feats, esm_feats, labels):
        self.struct_feats = struct_feats
        self.esm_feats = esm_feats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.struct_feats[idx], dtype=torch.float32),
            torch.tensor(self.esm_feats[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def extract_esm_features(sequences, model_name="esm2_t33_650M_UR50D", device=None, gpu=None):
    """
    使用ESM-2模型从蛋白质序列中提取特征

    Args:
        sequences: 蛋白质序列列表
        model_name: ESM-2模型名称
        device: 计算设备(cpu或cuda)
        gpu: GPU设备ID

    Returns:
        特征字典，键为序列索引，值为特征向量
    """
    if not ESM_AVAILABLE:
        logger.error("ESM-2模块未安装，无法提取特征")
        return None

    logger.info(f"使用ESM-2模型 {model_name} 提取特征")

    try:
        # 设置设备
        if gpu is not None:
            device_str = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(device_str)
        logger.info(f"ESM特征提取使用设备: {device}")

        # 加载ESM模型
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device)
        model.eval()

        batch_converter = alphabet.get_batch_converter()

        features = {}
        # 分批处理序列以避免内存溢出
        batch_size = 4  # 可根据GPU内存调整

        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_labels = [(f"seq_{i + j}", seq) for j, seq in enumerate(batch_sequences)]

            try:
                batch_tokens = batch_converter(batch_labels)[2].to(device)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])

                # 提取最后一层的特征
                token_representations = results["representations"][33]

                # 对每个序列进行处理
                for j, seq in enumerate(batch_sequences):
                    # 去除开始和结束标记的表示
                    seq_len = len(seq)
                    seq_repr = token_representations[j, 1:seq_len + 1, :].mean(dim=0).cpu().numpy()
                    features[i + j] = seq_repr

                logger.info(f"已处理 {min(i + batch_size, len(sequences))}/{len(sequences)} 个序列")

                # 清理GPU内存
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"处理批次 {i} 时出错: {e}")
                continue

        logger.info(f"已提取 {len(features)} 个序列的ESM-2特征")
        return features

    except Exception as e:
        logger.error(f"ESM-2特征提取失败: {e}")
        return None


def load_ids_labels_from_fasta(fasta_path):
    ids = []
    labels = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seq_id, label, _ = record.id.split('|')
        ids.append(seq_id)
        labels.append(int(label))
    return ids, np.array(labels)


def load_feature_dict(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    print(f"[DEBUG] {csv_path} 列名: {list(df.columns)}")
    # 只去除'sequence'列，保留peptide_id做索引
    if 'sequence' in df.columns:
        df = df.drop(columns=['sequence'])
    # 只保留数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    return {str(idx): row.values for idx, row in df.iterrows()}


def load_train_val_features_and_labels():
    # 路径
    train_fasta = 'Data/split_dataset/train_set.fasta'
    val_fasta = 'Data/split_dataset/val_set.fasta'
    struct_csv = 'Data/structure_features/structure_features.csv'
    esm_csv = 'Data/esm_features/esm_features.csv'

    # 读取ID和标签
    train_ids, y_train = load_ids_labels_from_fasta(train_fasta)
    val_ids, y_val = load_ids_labels_from_fasta(val_fasta)

    # 读取特征
    struct_feats = load_feature_dict(struct_csv)
    esm_feats = load_feature_dict(esm_csv)

    # 对齐特征
    X_struct_train = np.array([struct_feats[i] for i in train_ids if i in struct_feats and i in esm_feats],
                              dtype=np.float32)
    X_esm_train = np.array([esm_feats[i] for i in train_ids if i in struct_feats and i in esm_feats], dtype=np.float32)
    y_train = np.array([y for i, y in zip(train_ids, y_train) if i in struct_feats and i in esm_feats], dtype=np.int64)

    X_struct_val = np.array([struct_feats[i] for i in val_ids if i in struct_feats and i in esm_feats],
                            dtype=np.float32)
    X_esm_val = np.array([esm_feats[i] for i in val_ids if i in struct_feats and i in esm_feats], dtype=np.float32)
    y_val = np.array([y for i, y in zip(val_ids, y_val) if i in struct_feats and i in esm_feats], dtype=np.int64)

    return X_struct_train, X_esm_train, y_train, X_struct_val, X_esm_val, y_val, train_ids, val_ids


def load_features_and_labels(test_feature_file, train_feature_file, train_label_file, test_label_file, use_esm=False,
                             use_structure=True, use_raw_sequence=False, esm_model="esm2_t33_650M_UR50D",
                             projection_dim=512):
    """
    加载特征和标签数据，并使用投影方法融合特征

    Args:
        test_feature_file: 测试集特征文件路径
        train_feature_file: 训练集特征文件路径
        train_label_file: 训练集标签文件路径
        test_label_file: 测试集标签文件路径
        use_esm: 是否使用ESM-2提取额外特征
        use_structure: 是否使用结构特征
        use_raw_sequence: 是否直接使用氨基酸序列数据供LSTM处理
        esm_model: ESM-2模型名称
        projection_dim: 投影维度，默认为512。这个值越大，保留的特征信息越多，但计算成本也越高

    Returns:
        X_orig: 原始特征矩阵
        X_esm: ESM特征矩阵
        y: 标签向量
        peptide_ids: 肽ID列表
        all_df: 合并的标签数据框
        feature_dims: 特征维度信息的字典
    """
    # 初始化特征字典
    features_dict = {}

    # 加载测试集特征
    logger.info(f"加载测试集特征文件: {test_feature_file}")
    try:
        with open(test_feature_file, 'rb') as f:
            test_features = pickle.load(f)
        logger.info(f"成功加载测试集特征, 共有{len(test_features)}个肽")
        # 将测试特征添加到合并字典
        features_dict.update(test_features)
    except Exception as e:
        logger.error(f"加载测试集特征文件失败: {e}")
        sys.exit(1)

    # 加载训练集特征
    logger.info(f"加载训练集特征文件: {train_feature_file}")
    try:
        with open(train_feature_file, 'rb') as f:
            train_features = pickle.load(f)
        logger.info(f"成功加载训练集特征, 共有{len(train_features)}个肽")
        # 将训练特征添加到合并字典
        features_dict.update(train_features)
    except Exception as e:
        logger.error(f"加载训练集特征文件失败: {e}")
        sys.exit(1)

    logger.info(f"合并后共有{len(features_dict)}个肽的特征")

    # 加载训练集标签
    logger.info(f"加载训练集标签: {train_label_file}")
    try:
        train_df = pd.read_csv(train_label_file)
        logger.info(f"成功加载训练集标签, 共有{len(train_df)}条记录")
    except Exception as e:
        logger.error(f"加载训练集标签失败: {e}")
        sys.exit(1)

    # 加载测试集标签
    logger.info(f"加载测试集标签: {test_label_file}")
    try:
        test_df = pd.read_csv(test_label_file)
        logger.info(f"成功加载测试集标签, 共有{len(test_df)}条记录")
    except Exception as e:
        logger.error(f"加载测试集标签失败: {e}")
        sys.exit(1)

    # 合并训练集和测试集
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f"合并后共有{len(all_df)}条记录")

    # 获取列名(第一列是ID，第二列是蛋白质序列，第三列是标签)
    id_col = all_df.columns[0]  # ID列
    seq_col = all_df.columns[1]  # 序列列
    label_col = all_df.columns[2]  # 标签列

    logger.info(f"ID列: {id_col}, 序列列: {seq_col}, 标签列: {label_col}")

    # 如果需要使用ESM提取特征
    esm_features = None
    if use_esm and ESM_AVAILABLE:
        # 获取所有序列
        sequences = all_df[seq_col].tolist()
        logger.info(f"使用ESM-2提取额外特征，共有{len(sequences)}个序列")

        # 提取ESM特征
        # 创建索引到序列的映射，以便后续查找
        idx_to_seq = {i: seq for i, seq in enumerate(sequences)}
        # 提取ESM特征（使用索引作为键）
        esm_features = extract_esm_features(sequences, model_name=esm_model)

        if esm_features is None:
            logger.warning("ESM-2特征提取失败，将仅使用原始特征")

    # 设置设备为cuda:1，如果不可用则用cpu
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
        print('Using device: cuda:1')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using device: cuda:0')
    else:
        device = torch.device('cpu')
        print('Using device: cpu')
    # 训练和验证指标模板
    train_metrics = {
        'loss': [], 'acc': [], 'auc': [],
        'precision': [], 'recall': [], 'f1': []
    }
    val_metrics = {
        'loss': [], 'acc': [], 'auc': [],
        'precision': [], 'recall': [], 'f1': []
    }
    # 合并训练集和验证集用于交叉验证
    X_struct_all = np.concatenate([X_struct_train, X_struct_val], axis=0)
    X_esm_all = np.concatenate([X_esm_train, X_esm_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    all_ids = train_ids + val_ids

    # ====== 保存融合特征和标签（投影后）到pkl，便于可视化分析 ======
    import pickle
    feature_proj = FeatureProjection(X_struct_all.shape[1], X_esm_all.shape[1], output_dim=512)
    feature_proj.eval()
    with torch.no_grad():
        fused_features = feature_proj(
            torch.tensor(X_struct_all, dtype=torch.float32),
            torch.tensor(X_esm_all, dtype=torch.float32)
        ).numpy()
    os.makedirs('Data/split_dataset', exist_ok=True)
    with open('Data/split_dataset/balance_bak/fused_features.pkl', 'wb') as f:
        pickle.dump({'features': fused_features, 'labels': y_all}, f)
    print('融合特征已保存为 Data/split_dataset/fused_features.pkl')

    # 准备特征和标签数据
    X_orig_list = []  # 原始特征列表
    X_esm_list = []  # ESM特征列表
    y_list = []  # 标签列表
    peptide_ids = []  # 肽ID列表
    skipped_count = 0  # 跳过的记录数
    feature_dims = {}  # 记录特征维度

    # 对每个肽ID处理
    for idx, row in all_df.iterrows():
        peptide_id = str(row[id_col])  # 确保是字符串格式
        sequence = row[seq_col]  # 获取序列信息
        label = row[label_col]

        # 检查特征是否存在
        if peptide_id in features_dict:
            # 获取结构特征
            peptide_feature = features_dict[peptide_id]

            # 特征处理：提取统一特征表示(均值池化)
            # 如果特征是残基级别的，我们将其平均为肽级别的特征
            mean_feature = np.mean(peptide_feature, axis=0).astype(np.float32)
            X_orig_list.append(mean_feature)

            # 如果有ESM特征，添加到ESM特征列表
            if esm_features is not None and idx in esm_features:
                esm_feature = esm_features[idx].astype(np.float32)
                X_esm_list.append(esm_feature)

                # 记录特征维度
                if not feature_dims:
                    feature_dims['orig_dim'] = mean_feature.shape[0]
                    feature_dims['esm_dim'] = esm_feature.shape[0]
                    feature_dims['projection_dim'] = projection_dim
                    logger.info(f"原始特征维度: {mean_feature.shape[0]}")
                    logger.info(f"ESM特征维度: {esm_feature.shape[0]}")
                    compression_ratio_esm = esm_feature.shape[0] / projection_dim
                    compression_ratio_orig = mean_feature.shape[0] / projection_dim
                    logger.info(f"投影维度: {projection_dim}")
                    logger.info(f"压缩比例 - ESM: {compression_ratio_esm:.2f}x, 原始: {compression_ratio_orig:.2f}x")
                    logger.info(f"最终特征维度: {projection_dim * 2} (投影后拼接)")
            else:
                # 如果没有ESM特征，添加一个空特征
                if use_esm and esm_features is not None:
                    logger.warning(f"肽ID {peptide_id} 没有ESM特征，将跳过")
                    continue

            y_list.append(label)
            peptide_ids.append(peptide_id)
        else:
            skipped_count += 1

    if skipped_count > 0:
        logger.warning(f"有{skipped_count}个肽ID在特征文件中未找到，已跳过")

    # 转换为numpy数组，确保float32类型
    X_orig = np.array(X_orig_list, dtype=np.float32)
    X_esm = np.array(X_esm_list, dtype=np.float32) if X_esm_list else None
    y = np.array(y_list)

    # 根据用户选择判断是否使用结构特征
    if not use_structure:
        if use_raw_sequence:
            logger.info(f"禁用结构特征，将直接使用氨基酸序列进行LSTM处理")

            # 创建氨基酸到索引的映射
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20种标准氨基酸
            aa_to_idx = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # 从1开始编号，0留给填充符
            aa_to_idx['<PAD>'] = 0  # 填充符
            aa_to_idx['X'] = 20  # 未知氨基酸

            # 获取所有序列的最大长度，用于填充
            max_seq_length = 0
            peptide_sequences = []

            for peptide_id in peptide_ids:
                seq = all_df[all_df[id_col] == peptide_id][seq_col].values[0]
                peptide_sequences.append(seq)
                max_seq_length = max(max_seq_length, len(seq))

            logger.info(f"最大序列长度: {max_seq_length}")

            # 将氨基酸序列转换为索引序列并填充
            sequence_indices = []
            for seq in peptide_sequences:
                # 将序列转换为索引
                indices = [aa_to_idx.get(aa, 20) for aa in seq]  # 未知氨基酸用20表示

                # 填充到最大长度
                if len(indices) < max_seq_length:
                    indices = indices + [0] * (max_seq_length - len(indices))

                sequence_indices.append(indices)

            # 创建特征矩阵
            X_orig = np.array(sequence_indices, dtype=np.int64)
            logger.info(f"序列索引矩阵形状: {X_orig.shape}")

            # 设置标志以启用嵌入层
            feature_dims['use_embedding'] = True
            feature_dims['vocab_size'] = 21  # 20种氨基酸 + 1个填充符
            feature_dims['max_seq_length'] = max_seq_length
        else:
            logger.info(f"禁用结构特征，将仅使用ESM特征或空特征")
            # 创建一个占位特征矩阵，保持样本数不变
            X_orig = np.zeros((X_orig.shape[0], 1), dtype=np.float32)

    # 只在禁用结构特征时添加该标志，保持默认情况下完全兼容
    if not use_structure:
        feature_dims['use_structure'] = False

    # 确保即使只使用单一特征也创建投影层
    if not use_raw_sequence:
        # 如果仅使用结构特征，为其创建一个空的ESM特征矩阵
        if use_structure and not use_esm:
            logger.info("仅使用结构特征，但仍将通过投影层处理以保持一致性")
            # 创建一个与结构特征相同样本数的伪ESM特征，维度为projection_dim
            # 这样投影层仍然可以工作，但只有结构特征有意义
            X_esm = np.zeros((X_orig.shape[0], projection_dim), dtype=np.float32)
            feature_dims['esm_dim'] = projection_dim

        # 如果仅使用ESM特征，为其创建一个空的结构特征矩阵
        elif use_esm and not use_structure:
            logger.info("仅使用ESM特征，但仍将通过投影层处理以保持一致性")
            # 创建一个与ESM特征相同样本数的伪结构特征，维度为projection_dim
            # 这样投影层仍然可以工作，但只有ESM特征有意义
            X_orig = np.zeros((X_esm.shape[0], projection_dim), dtype=np.float32)
            feature_dims['orig_dim'] = projection_dim

    return X_orig, X_esm, y, peptide_ids, all_df, feature_dims


def train_epoch(model, feature_proj, train_loader, criterion, optimizer, device):
    model.train()
    feature_proj.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for struct_x, esm_x, labels in tqdm(train_loader, desc="Training"):
        struct_x, esm_x, labels = struct_x.to(device), esm_x.to(device), labels.to(device)
        optimizer.zero_grad()
        fused = feature_proj(struct_x, esm_x).unsqueeze(1)
        logits = model(fused)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
    train_acc = train_correct / train_total
    train_loss /= train_total
    return train_loss, train_acc


def validate_epoch(model, feature_proj, val_loader, criterion, device):
    model.eval()
    feature_proj.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_labels_all = []
    val_preds_all = []
    val_probs_all = []
    with torch.no_grad():
        for struct_x, esm_x, labels in tqdm(val_loader, desc="Validation"):
            struct_x, esm_x, labels = struct_x.to(device), esm_x.to(device), labels.to(device)
            fused = feature_proj(struct_x, esm_x).unsqueeze(1)
            logits = model(fused)
            loss = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_labels_all.extend(labels.cpu().numpy())
            val_preds_all.extend(preds.cpu().numpy())
            val_probs_all.extend(probs.cpu().numpy())
    val_acc = val_correct / val_total
    val_loss /= val_total
    return val_loss, val_acc, val_labels_all, val_preds_all, val_probs_all


def evaluate_model(y_true, y_pred, y_proba):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float('nan')
    return precision, recall, f1, specificity, mcc, auc


def plot_metrics(train_metrics, val_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['loss'], label='Train Loss')
    plt.plot(val_metrics['loss'], label='Val Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['acc'], label='Train Acc')
    plt.plot(val_metrics['acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['auc'], label='Train AUC')
    plt.plot(val_metrics['auc'], label='Val AUC')
    plt.title('Training and Validation AUC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'auc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_evaluation_curves(y_true, y_proba, save_dir):
    """
    绘制ROC曲线和PR曲线

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_predictions(peptide_ids, sequences, y_true, y_pred, y_proba, save_path):
    results_df = pd.DataFrame({
        'peptide_id': peptide_ids,
        'sequence': sequences,
        'true_label': y_true,
        'predicted_label': y_pred,
        'prediction_probability': y_proba
    })
    results_df.to_csv(save_path, index=False)


def cross_validate(X_struct_all, X_esm_all, y_all, all_ids, id_to_seq, config, device, exp_dir):
    # 训练和验证指标模板
    train_metrics = {
        'loss': [], 'acc': [], 'auc': [],
        'precision': [], 'recall': [], 'f1': []
    }
    val_metrics = {
        'loss': [], 'acc': [], 'auc': [],
        'precision': [], 'recall': [], 'f1': []
    }

    # 初始化模型预测概率收集字典
    model_confidences = {'LSTM': []}
    model_labels = {'LSTM': []}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['random_seed'])
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_struct_all, y_all)):
        print(f"\n========== Fold {fold + 1}/5 ==========")
        train_dataset = FusionDataset(X_struct_all[train_idx], X_esm_all[train_idx], y_all[train_idx])
        val_dataset = FusionDataset(X_struct_all[val_idx], X_esm_all[val_idx], y_all[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training_params']['batch_size'])

        feature_proj = FeatureProjection(X_struct_all.shape[1], X_esm_all.shape[1], output_dim=512).to(device)
        classifier = LSTMClassifier(
            input_dim=1024,
            hidden_dim=config['model_params']['hidden_dim'],
            num_layers=config['model_params']['num_layers'],
            dropout=config['model_params']['dropout'],
            num_classes=config['model_params']['num_classes']
        ).to(device)
        optimizer = torch.optim.Adam(
            list(feature_proj.parameters()) + list(classifier.parameters()),
            lr=config['training_params']['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        best_state = None
        best_metrics = None
        current_train_metrics = {k: [] for k in train_metrics.keys()}
        current_val_metrics = {k: [] for k in val_metrics.keys()}
        patience = config['training_params']['patience']
        patience_counter = 0

        fold_dir = os.path.join(exp_dir, f'fold_{fold + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        csv_path = os.path.join(fold_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'Val_Precision', 'Val_Recall', 'Val_F1',
                 'Val_Specificity', 'Val_MCC', 'Val_AUC'])

        for epoch in range(config['training_params']['num_epochs']):
            train_loss, train_acc = train_epoch(classifier, feature_proj, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_labels_all, val_preds_all, val_probs_all = validate_epoch(classifier, feature_proj,
                                                                                             val_loader, criterion,
                                                                                             device)
            precision, recall, f1, specificity, mcc, auc = evaluate_model(val_labels_all, val_preds_all, val_probs_all)

            print(
                f"Epoch {epoch + 1}/{config['training_params']['num_epochs']} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(
                f"                 | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val_Precision: {precision:.4f} | Val_Recall: {recall:.4f} | Val_F1: {f1:.4f} | Val_Specificity: {specificity:.4f} | Val_MCC: {mcc:.4f} | Val_AUC: {auc:.4f}")

            current_train_metrics['loss'].append(train_loss)
            current_train_metrics['acc'].append(train_acc)
            current_train_metrics['auc'].append(auc)
            current_train_metrics['precision'].append(precision)
            current_train_metrics['recall'].append(recall)
            current_train_metrics['f1'].append(f1)
            current_val_metrics['loss'].append(val_loss)
            current_val_metrics['acc'].append(val_acc)
            current_val_metrics['auc'].append(auc)
            current_val_metrics['precision'].append(precision)
            current_val_metrics['recall'].append(recall)
            current_val_metrics['f1'].append(f1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'feature_proj': feature_proj.state_dict(),
                    'classifier': classifier.state_dict()
                }
                best_metrics = {
                    'Val_Loss': val_loss,
                    'Val_Acc': val_acc,
                    'Val_Precision': precision,
                    'Val_Recall': recall,
                    'Val_F1': f1,
                    'Val_Specificity': specificity,
                    'Val_MCC': mcc,
                    'Val_AUC': auc
                }
                # 保存最佳模型状态时的验证集预测概率
                best_val_probs = val_probs_all
                torch.save(best_state, os.path.join(fold_dir, f'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"[EarlyStopping] No improvement for {patience} epochs, stopping training.")
                break

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [epoch + 1, train_loss, train_acc, val_loss, val_acc, precision, recall, f1, specificity, mcc, auc])

        # 将最佳模型状态下的验证集预测概率和标签添加到收集字典中
        model_confidences['LSTM'].append(best_val_probs)
        model_labels['LSTM'].append(val_labels_all)

        print(f"Fold {fold + 1} metrics saved to {csv_path}")

        plot_metrics(current_train_metrics, current_val_metrics, fold_dir)
        plot_evaluation_curves(val_labels_all, val_probs_all, fold_dir)
        plot_confusion_matrix(val_labels_all, val_preds_all, fold_dir)
        save_predictions(
            [all_ids[i] for i in val_idx],
            [id_to_seq.get(all_ids[i], "") for i in val_idx],
            val_labels_all,
            val_preds_all,
            val_probs_all,
            os.path.join(fold_dir, 'prediction_results.csv')
        )
        all_metrics.append(best_metrics)

    # 计算并保存平均指标
    avg_metrics_dir = os.path.join(exp_dir, 'average_metrics')
    os.makedirs(avg_metrics_dir, exist_ok=True)
    keys = all_metrics[0].keys()
    rows = []
    for k in keys:
        values = [m[k] for m in all_metrics]
        rows.append({
            'Metric': k,
            'Mean': np.mean(values),
            'Std': np.std(values)
        })
    avg_metrics_df = pd.DataFrame(rows)
    avg_metrics_df.to_csv(os.path.join(avg_metrics_dir, 'average_metrics.csv'), index=False)

    # 保存最佳模型
    best_idx = int(np.argmax([m['Val_Acc'] for m in all_metrics]))
    best_model_path = os.path.join(exp_dir, f'fold_{best_idx + 1}', 'best_model.pth')
    shutil.copy(best_model_path, os.path.join(exp_dir, 'best_model.pth'))
    print(f"Best model has been saved as: {os.path.join(exp_dir, 'best_model.pth')}")
    print(f"Best model's Val Acc: {all_metrics[best_idx]['Val_Acc']:.4f}")
    print("Best model's all metrics:")
    for k, v in all_metrics[best_idx].items():
        print(f"  {k}: {v:.4f}")

    # 绘制验证集正负样本预测概率密度分布（左右两个子图）
    # 合并所有fold的概率和标签
    model_pos_probs = {}
    model_neg_probs = {}
    for model in model_confidences:
        if model.endswith('_pos') or model.endswith('_neg'):
            continue
        all_probs = [p for fold_probs in model_confidences[model] for p in fold_probs]
        all_labels = [l for fold_labels in model_labels[model] for l in fold_labels]
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        pos_probs = all_probs[all_labels == 1]
        neg_probs = all_probs[all_labels == 0]
        model_pos_probs[model] = pos_probs
        model_neg_probs[model] = neg_probs

    plt.figure(figsize=(14, 6))
    # 左图：正样本
    plt.subplot(1, 2, 1)
    for model in model_pos_probs:
        if len(model_pos_probs[model]) > 1:
            sns.kdeplot(model_pos_probs[model], label=f'{model} Positive', fill=True, alpha=0.3)
    plt.title('Density (predicted positive samples)')
    plt.xlabel('Prediction Confidence (Probability)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    # 右图：负样本
    plt.subplot(1, 2, 2)
    for model in model_neg_probs:
        if len(model_neg_probs[model]) > 1:
            sns.kdeplot(model_neg_probs[model], label=f'{model} Negative', fill=True, alpha=0.3)
    plt.title('Density (predicted negative samples)')
    plt.xlabel('Prediction Confidence (Probability)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'validation_confidence_density_pos_neg.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存预测概率数据
    for model, probs_list in model_confidences.items():
        all_probs = [p for fold_probs in probs_list for p in fold_probs]
        np.save(os.path.join(exp_dir, f'{model}_validation_probs.npy'), np.array(all_probs))


def main():
    """
    Main function for running LSTM classifier training and evaluation
    """
    # 设置随机种子
    set_all_seeds(42)

    # 生成实验ID和实验目录
    import time
    exp_id = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(os.getcwd(), 'Results', 'experiments', exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    # 保存实验配置
    config = {
        'random_seed': 42,
        'model_params': {
            'input_dim': 1024,
            'hidden_dim': 128,
            'num_layers': 1,
            'dropout': 0.2,
            'num_classes': 2
        },
        'training_params': {
            'batch_size': 32,
            'num_epochs': 30,
            'learning_rate': 1e-3,
            'patience': 5
        },
        'data_paths': {
            'train_fasta': 'Data/split_dataset/train_set.fasta',
            'val_fasta': 'Data/split_dataset/val_set.fasta',
            'struct_csv': 'Data/structure_features/structure_features.csv',
            'esm_csv': 'Data/esm_features/esm_features.csv'
        }
    }
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Experiment configuration saved to {os.path.join(exp_dir, 'config.json')}")

    # 设置随机种子
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])

    # 直接用新函数加载数据
    X_struct_train, X_esm_train, y_train, X_struct_val, X_esm_val, y_val, train_ids, val_ids = load_train_val_features_and_labels()

    # Build id_to_seq mapping for val_ids
    val_fasta = config['data_paths']['val_fasta']
    id_to_seq = {}
    for record in SeqIO.parse(val_fasta, 'fasta'):
        seq_id, label, _ = record.id.split('|')
        id_to_seq[seq_id] = str(record.seq)

    print(f"Structure feature dimension: {X_struct_train.shape[1]}")
    print(f"ESM feature dimension: {X_esm_train.shape[1]}")
    print(f"Training set size: {X_struct_train.shape[0]}")
    print(f"Validation set size: {X_struct_val.shape[0]}")

    # 设置设备为cuda:1，如果不可用则用cpu
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
        print('Using device: cuda:1')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using device: cuda:0')
    else:
        device = torch.device('cpu')
        print('Using device: cpu')

    # ====== 五折交叉验证只在训练集上做，验证集单独评估 ======
    # 1. 训练集和验证集保持分离
    X_struct_train = X_struct_train
    X_esm_train = X_esm_train
    y_train = y_train
    train_ids = train_ids

    # 2. 五折交叉验证（只用训练集）
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    output_dim = 512

    # 存储每一折在验证集上的性能
    fold_val_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_struct_train, y_train)):
        print(f"\n========== Fold {fold + 1}/5 ==========")
        X_tr_struct, X_tr_esm, y_tr = X_struct_train[train_idx], X_esm_train[train_idx], y_train[train_idx]
        X_va_struct, X_va_esm, y_va = X_struct_train[val_idx], X_esm_train[val_idx], y_train[val_idx]

        # 构建Dataset和DataLoader
        class FusionDataset(Dataset):
            def __init__(self, struct_feats, esm_feats, labels):
                self.struct_feats = torch.FloatTensor(struct_feats)
                self.esm_feats = torch.FloatTensor(esm_feats)
                self.labels = torch.LongTensor(labels)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.struct_feats[idx], self.esm_feats[idx], self.labels[idx]

        train_dataset = FusionDataset(X_tr_struct, X_tr_esm, y_tr)
        val_dataset = FusionDataset(X_va_struct, X_va_esm, y_va)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # 初始化模型
        feature_proj = FeatureProjection(X_tr_struct.shape[1], X_tr_esm.shape[1], output_dim=output_dim).to(device)
        classifier = LSTMClassifier(
            input_dim=output_dim * 2,
            hidden_dim=config['model_params']['hidden_dim'],
            num_layers=config['model_params']['num_layers'],
            dropout=config['model_params']['dropout'],
            num_classes=config['model_params']['num_classes']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(feature_proj.parameters()) + list(classifier.parameters()), lr=1e-3)
        num_epochs = 30

        # 训练模型
        best_fold_val_acc = 0
        best_fold_feature_proj_state = None
        best_fold_classifier_state = None

        for epoch in range(num_epochs):
            feature_proj.train()
            classifier.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for struct_x, esm_x, yb in train_loader:
                struct_x, esm_x, yb = struct_x.to(device), esm_x.to(device), yb.to(device)
                optimizer.zero_grad()
                fused = feature_proj(struct_x, esm_x)
                logits = classifier(fused)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * yb.size(0)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == yb).sum().item()
                train_total += yb.size(0)

            train_acc = train_correct / train_total
            train_loss /= train_total

            # 在训练集内部验证
            feature_proj.eval()
            classifier.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            val_probs, val_labels, val_preds = [], [], []

            with torch.no_grad():
                for struct_x, esm_x, yb in val_loader:
                    struct_x, esm_x, yb = struct_x.to(device), esm_x.to(device), yb.to(device)
                    fused = feature_proj(struct_x, esm_x)
                    logits = classifier(fused)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * yb.size(0)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_probs.extend(probs)
                    val_preds.extend(preds)
                    val_labels.extend(yb.cpu().numpy())
                    val_correct += (preds == yb.cpu().numpy()).sum()
                    val_total += yb.size(0)

            val_acc = val_correct / val_total
            val_loss /= val_total

            if val_acc > best_fold_val_acc:
                best_fold_val_acc = val_acc
                best_fold_feature_proj_state = feature_proj.state_dict()
                best_fold_classifier_state = classifier.state_dict()

            print(
                f"Fold {fold + 1} Epoch {epoch + 1:02d}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        # 3. 用当前折的最优模型在独立验证集上评估
        feature_proj_val = FeatureProjection(X_struct_val.shape[1], X_esm_val.shape[1], output_dim=output_dim).to(
            device)
        classifier_val = LSTMClassifier(
            input_dim=output_dim * 2,
            hidden_dim=config['model_params']['hidden_dim'],
            num_layers=config['model_params']['num_layers'],
            dropout=config['model_params']['dropout'],
            num_classes=config['model_params']['num_classes']
        ).to(device)

        feature_proj_val.load_state_dict(best_fold_feature_proj_state)
        classifier_val.load_state_dict(best_fold_classifier_state)
        feature_proj_val.eval()
        classifier_val.eval()

        class FusionDatasetVal(Dataset):
            def __init__(self, struct_feats, esm_feats, labels):
                self.struct_feats = torch.FloatTensor(struct_feats)
                self.esm_feats = torch.FloatTensor(esm_feats)
                self.labels = torch.LongTensor(labels)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.struct_feats[idx], self.esm_feats[idx], self.labels[idx]

        val_dataset_independent = FusionDatasetVal(X_struct_val, X_esm_val, y_val)
        val_loader_independent = DataLoader(val_dataset_independent, batch_size=32)

        val_probs_independent, val_labels_independent, val_preds_independent = [], [], []

        with torch.no_grad():
            for struct_x, esm_x, yb in val_loader_independent:
                struct_x, esm_x, yb = struct_x.to(device), esm_x.to(device), yb.to(device)
                fused = feature_proj_val(struct_x, esm_x)
                logits = classifier_val(fused)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_probs_independent.extend(probs)
                val_preds_independent.extend(preds)
                val_labels_independent.extend(yb.cpu().numpy())

        # 计算当前折在独立验证集上的性能
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
            matthews_corrcoef, confusion_matrix
        acc = accuracy_score(val_labels_independent, val_preds_independent)
        precision = precision_score(val_labels_independent, val_preds_independent)
        recall = recall_score(val_labels_independent, val_preds_independent)
        f1 = f1_score(val_labels_independent, val_preds_independent)
        auc = roc_auc_score(val_labels_independent, val_probs_independent)
        mcc = matthews_corrcoef(val_labels_independent, val_preds_independent)

        # 计算混淆矩阵以获得SP（特异性）和SN（敏感性）
        cm = confusion_matrix(val_labels_independent, val_preds_independent)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异性
            sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 敏感性/召回率
        else:
            sp = 0.0
            sn = 0.0

        fold_metrics = {
            'fold': fold + 1,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'mcc': mcc,
            'sp': sp,
            'sn': sn
        }
        fold_val_metrics.append(fold_metrics)

        print(
            f"Fold {fold + 1} 在独立验证集上的性能：Acc={acc:.4f} Precision={precision:.4f} Recall={recall:.4f} SP={sp:.4f} F1={f1:.4f} AUC={auc:.4f} MCC={mcc:.4f}")

    # 4. 计算五折交叉验证的平均性能
    print(f"\n========== 五折交叉验证平均性能 ==========")
    avg_metrics = {}
    for metric in ['acc', 'precision', 'recall', 'f1', 'auc', 'mcc', 'sp', 'sn']:
        values = [fold_metrics[metric] for fold_metrics in fold_val_metrics]
        avg_metrics[metric] = np.mean(values)
        std_metrics = np.std(values)
        print(f"{metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics:.4f}")

    # 5. 保存五折交叉验证结果
    import pandas as pd
    fold_results_df = pd.DataFrame(fold_val_metrics)
    fold_results_df.to_csv(os.path.join(exp_dir, 'fold_cv_results.csv'), index=False)

    # 保存平均性能
    avg_results_df = pd.DataFrame([avg_metrics])
    avg_results_df.to_csv(os.path.join(exp_dir, 'cv_average_results.csv'), index=False)

    print(f"五折交叉验证详细结果已保存到: {os.path.join(exp_dir, 'fold_cv_results.csv')}")
    print(f"五折交叉验证平均性能已保存到: {os.path.join(exp_dir, 'cv_average_results.csv')}")

    # ====== 保存最后一折的模型参数作为示例 ======
    torch.save({
        'feature_proj': best_fold_feature_proj_state,
        'classifier': best_fold_classifier_state
    }, os.path.join(exp_dir, 'last_fold_model.pth'))
    print(f"最后一折模型参数已保存到: {os.path.join(exp_dir, 'last_fold_model.pth')}")


if __name__ == "__main__":
    main()
