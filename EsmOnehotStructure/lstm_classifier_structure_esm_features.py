#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质序列LSTM分类器
该脚本从预处理的特征文件中加载蛋白质特征数据，
并使用ESM-2预训练模型提取额外的特征，
然后通过投影层将不同特征投影至相同维度后融合，
最后使用LSTM神经网络进行分类，
并进行5折交叉验证评估模型性能。

作者：JinjinLi
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置matplotlib使用支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                                       'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"设置中文字体时出错: {e}, 可能导致中文显示异常")

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
            x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # LSTM处理: [batch_size, seq_len, input_dim/embedding_dim]
        # LSTM输出: output - [batch_size, seq_len, hidden_dim*2]
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
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


def train_lstm_model(model, feature_projection, X_orig, X_esm, train_indices, val_indices, y,
                     batch_size=16, num_epochs=30, device='cuda', learning_rate=0.001, patience=5):
    """
    训练使用投影融合特征的LSTM模型

    Args:
        model: LSTM模型
        feature_projection: 特征投影层
        X_orig: 原始特征矩阵
        X_esm: ESM特征矩阵
        train_indices: 训练集指标
        val_indices: 验证集指标
        y: 标签向量
        batch_size: 批次大小
        num_epochs: 训练轮数
        device: 训练设备
        learning_rate: 学习率
        patience: 提前停止的耐心值

    Returns:
        训练好的模型，特征投影层和最佳验证损失值
    """
    # 将模型移至设备
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 创建参数列表，根据是否使用投影层决定要优化的参数
    if feature_projection is not None:
        feature_projection = feature_projection.to(device)
        params = list(model.parameters()) + list(feature_projection.parameters())
    else:
        params = model.parameters()

    optimizer = optim.Adam(params, lr=learning_rate)

    # 用于提前停止
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_projection_state = None

    # 判断是否是序列数据（使用嵌入层）
    is_sequence_data = hasattr(model, 'use_embedding') and model.use_embedding

    # 准备一个类来封装可变长度的批次
    class VariableLengthDataLoader:
        def __init__(self, X_orig, X_esm, y, indices, batch_size, shuffle=True, is_sequence=False):
            self.X_orig = X_orig
            self.X_esm = X_esm
            self.y = y
            self.indices = indices
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.is_sequence = is_sequence
            self.n_samples = len(indices)
            self.batch_count = (self.n_samples + self.batch_size - 1) // self.batch_size
            self._generate_batches()

        def _generate_batches(self):
            # 生成批次指标
            indices = np.copy(self.indices)
            if self.shuffle:
                np.random.shuffle(indices)
            self.batches = [
                indices[i:i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]

        def __iter__(self):
            if self.shuffle:
                self._generate_batches()
            for batch_indices in self.batches:
                # 提取批次数据
                if self.is_sequence:
                    # 序列数据使用LongTensor
                    batch_X_orig = torch.tensor(self.X_orig[batch_indices], dtype=torch.long)
                else:
                    # 浮点特征使用FloatTensor
                    batch_X_orig = torch.tensor(self.X_orig[batch_indices], dtype=torch.float32)

                batch_y = torch.tensor(self.y[batch_indices], dtype=torch.long)

                if self.X_esm is not None:
                    batch_X_esm = torch.tensor(self.X_esm[batch_indices], dtype=torch.float32)
                    yield batch_X_orig, batch_X_esm, batch_y
                else:
                    yield batch_X_orig, None, batch_y

        def __len__(self):
            return self.batch_count

    # 创建训练和验证数据加载器
    train_loader = VariableLengthDataLoader(X_orig, X_esm, y, train_indices, batch_size,
                                            shuffle=True, is_sequence=is_sequence_data)
    val_loader = VariableLengthDataLoader(X_orig, X_esm, y, val_indices, batch_size,
                                          shuffle=False, is_sequence=is_sequence_data)

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        if feature_projection is not None:
            feature_projection.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_orig_batch, X_esm_batch, y_batch in train_loader:
            # 在每个batch前清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 将数据移至设备
            X_orig_batch = X_orig_batch.to(device)
            y_batch = y_batch.to(device)
            if X_esm_batch is not None:
                X_esm_batch = X_esm_batch.to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 如果是序列数据，使用嵌入层处理
            if is_sequence_data:
                outputs = model(X_orig_batch)
            else:
                # 投影特征
                if X_esm_batch is not None and feature_projection is not None:
                    # 使用投影层融合特征
                    projected_features = feature_projection(X_orig_batch, X_esm_batch)
                else:
                    # 如果没有ESM特征或没有投影层，直接使用原始特征
                    projected_features = X_orig_batch

                # 在LSTM模型中添加序列维度 (batch_size, 1, feature_dim)
                projected_features = projected_features.unsqueeze(1)

                # 前向传播
                outputs = model(projected_features)

            # 计算损失
            loss = criterion(outputs, y_batch)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        if feature_projection is not None:
            feature_projection.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_orig_batch, X_esm_batch, y_batch in val_loader:
                # 在每个batch前清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 将数据移至设备
                X_orig_batch = X_orig_batch.to(device)
                y_batch = y_batch.to(device)
                if X_esm_batch is not None:
                    X_esm_batch = X_esm_batch.to(device)

                # 如果是序列数据，使用嵌入层处理
                if is_sequence_data:
                    outputs = model(X_orig_batch)
                else:
                    # 投影特征
                    if X_esm_batch is not None and feature_projection is not None:
                        # 使用投影层融合特征
                        projected_features = feature_projection(X_orig_batch, X_esm_batch)
                    else:
                        # 如果没有ESM特征或没有投影层，直接使用原始特征
                        projected_features = X_orig_batch

                    # 在LSTM模型中添加序列维度 (batch_size, 1, feature_dim)
                    projected_features = projected_features.unsqueeze(1)

                    # 前向传播
                    outputs = model(projected_features)

                # 计算损失
                loss = criterion(outputs, y_batch)

                # 统计
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        # 计算验证指标
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # 打印进度
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 检查是否需要保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            if feature_projection is not None:
                best_projection_state = feature_projection.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # 提前停止
        if patience_counter >= patience:
            logger.info(f"提前停止训练，未见验证损失改进 {patience} 轮")
            break

    # 加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_projection_state is not None and feature_projection is not None:
        feature_projection.load_state_dict(best_projection_state)

    return model, feature_projection, best_val_loss


def evaluate_model(X_orig, X_esm, y, peptide_ids, all_df, feature_dims, device, output_dir, batch_size=16):
    """
    使用投影融合特征评估LSTM模型性能

    Args:
        X_orig: 原始特征矩阵
        X_esm: ESM特征矩阵
        y: 标签向量
        peptide_ids: 肽ID列表
        all_df: 包含序列信息的数据框
        feature_dims: 特征维度信息的字典
        device: 计算设备
        output_dir: 输出目录
        batch_size: 批次大小

    Returns:
        性能指标字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 保存每一折的指标
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],  # 等同于灵敏度(SN)
        'specificity': [],  # 特异性(SP)
        'f1': [],
        'mcc': [],  # Matthews相关系数
        'auc': [],
        'loss': []  # 验证集损失值
    }

    # 记录所有预测结果
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_test_indices = []

    # 保存最佳模型的变量
    best_f1 = -1
    best_model = None
    best_projection = None

    # 开始交叉验证
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_orig, y)):
        logger.info(f"开始第 {fold + 1} 折交叉验证")

        # 记录此折的测试集指标
        test_pids = [peptide_ids[i] for i in test_idx]

        # 获取投影维度
        projection_dim = feature_dims.get('projection_dim', 512)
        orig_dim = feature_dims.get('orig_dim', X_orig.shape[1])

        # 如果有ESM特征，获取ESM维度
        esm_dim = feature_dims.get('esm_dim', 0)

        # 检查是否使用嵌入层处理序列数据
        use_embedding = feature_dims.get('use_embedding', False)
        is_sequence = feature_dims.get('is_sequence', False)

        # 创建特征投影层
        if X_esm is not None and not is_sequence:
            feature_projection = FeatureProjection(orig_dim, esm_dim, projection_dim)
            input_dim = projection_dim * 2  # 投影后的拼接维度
        else:
            feature_projection = None
            if is_sequence:
                # 对于序列数据，输入维度等于嵌入维度
                input_dim = feature_dims.get('embedding_dim', 64)
            else:
                input_dim = orig_dim  # 原始特征维度

        # 创建模型
        hidden_dim = 64  # 可调整
        num_layers = 1

        # 创建LSTM分类器，对于序列数据启用嵌入层
        if use_embedding:
            vocab_size = feature_dims.get('vocab_size', 21)
            embedding_dim = feature_dims.get('embedding_dim', 64)
            fold_model = LSTMClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.2,
                num_classes=2,
                use_embedding=True,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim
            )
        else:
            fold_model = LSTMClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.2,
                num_classes=2
            )

        # 训练模型
        try:
            fold_model, fold_projection, best_val_loss = train_lstm_model(
                model=fold_model,
                feature_projection=feature_projection,
                X_orig=X_orig,
                X_esm=X_esm,
                train_indices=train_idx,
                val_indices=test_idx,  # 使用测试集作为验证集
                y=y,
                batch_size=batch_size,
                num_epochs=30,
                device=device,
                learning_rate=0.001,
                patience=5
            )
        except Exception as e:
            error_message = str(e)
            if "shape" in error_message or "dimension" in error_message:
                logger.error(f"维度不匹配错误！")
                if X_esm is not None:
                    logger.error(f"模型预期输入维度: {input_dim}，但数据维度不匹配。")
                    logger.error(
                        f"X_orig形状: {X_orig.shape}, X_esm形状: {X_esm.shape if X_esm is not None else 'None'}")
                logger.error(f"错误详情: {error_message}")
                logger.error(f"由于维度不匹配，跳过此折。")
                continue
            else:
                raise e

        # 评估模型
        fold_model.eval()
        if fold_projection is not None:
            fold_projection.eval()

        y_pred = []
        y_proba = []

        with torch.no_grad():
            # 每次计算一小批数据，避免内存问题
            for i in range(0, len(test_idx), batch_size):
                batch_indices = test_idx[i:i + batch_size]

                if use_embedding:
                    # 对于序列数据，直接使用LongTensor
                    X_test_batch = torch.LongTensor(X_orig[batch_indices]).to(device)
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 前向传播
                    outputs = fold_model(X_test_batch)
                else:
                    X_orig_batch = torch.FloatTensor(X_orig[batch_indices]).to(device)

                    if X_esm is not None:
                        X_esm_batch = torch.FloatTensor(X_esm[batch_indices]).to(device)
                        # 使用投影层
                        projected_features = fold_projection(X_orig_batch, X_esm_batch)
                    else:
                        # 如果没有ESM特征，直接使用原始特征
                        projected_features = X_orig_batch

                    # 在LSTM模型中添加序列维度 (batch_size, 1, feature_dim)
                    projected_features = projected_features.unsqueeze(1)

                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # 前向传播
                    outputs = fold_model(projected_features)

                proba = torch.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_proba.extend(proba[:, 1].cpu().numpy())  # 取正类的概率

        # 计算指标
        y_test = y[test_idx]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)  # 等同于灵敏度(SN)

        # 计算混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # 计算特异性(SP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 计算F1分数
        f1 = f1_score(y_test, y_pred)

        # 计算Matthews相关系数(MCC)
        mcc = matthews_corrcoef(y_test, y_pred)

        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            logger.warning(f"第 {fold + 1} 折无法计算AUC")
            auc = None

        # 保存指标
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)  # 等同于灵敏度(SN)
        fold_metrics['specificity'].append(specificity)  # 特异性(SP)
        fold_metrics['f1'].append(f1)
        fold_metrics['mcc'].append(mcc)  # Matthews相关系数
        fold_metrics['loss'].append(best_val_loss)  # 损失值
        if auc is not None:
            fold_metrics['auc'].append(auc)

        # 记录此折的结果
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_test_indices.extend(test_idx)

        # 输出当前折的结果
        logger.info(
            f"第 {fold + 1} 折 - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, "
            f"灵敏度(SN): {recall:.4f}, 特异性(SP): {specificity:.4f}, "
            f"F1: {f1:.4f}, MCC: {mcc:.4f}, Loss: {best_val_loss:.4f}" +
            (f", AUC: {auc:.4f}" if auc is not None else ""))

        # 判断这个模型是否是最好的（基于F1分数）
        if fold_metrics['f1'][-1] > best_f1:
            best_f1 = fold_metrics['f1'][-1]
            best_model = fold_model
            best_projection = fold_projection
            logger.info(f"找到新的最佳模型（第{fold+1}折），F1: {best_f1:.4f}")

    # 计算平均指标
    avg_metrics = {}
    for metric, values in fold_metrics.items():
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    # 输出交叉验证结果
    logger.info("交叉验证平均性能:")
    for metric, stats in avg_metrics.items():
        logger.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # 创建模型保存目录
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 保存最佳模型
    if best_model is not None:
        logger.info(f"保存最佳模型，F1: {best_f1:.4f}...")
        best_model_path = os.path.join(models_dir, 'best_model.pth')
        torch.save(best_model.state_dict(), best_model_path)

        # 如果有投影层，也保存投影层
        if best_projection is not None:
            best_projection_path = os.path.join(models_dir, 'best_projection.pth')
            torch.save(best_projection.state_dict(), best_projection_path)
            logger.info(f"最佳模型和投影层已保存至 {models_dir}")
        else:
            logger.info(f"最佳模型已保存至 {models_dir}")

    # 保存交叉验证结果
    cv_results = {
        'Metric': [],
        'Mean': [],
        'Std': []
    }

    for metric, stats in avg_metrics.items():
        cv_results['Metric'].append(metric.capitalize())
        cv_results['Mean'].append(stats['mean'])
        cv_results['Std'].append(stats['std'])

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)

    # 保存分类报告
    class_report = classification_report(all_y_true, all_y_pred)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    # 获取原始序列信息
    id_to_seq = {}
    id_col = all_df.columns[0]
    seq_col = all_df.columns[1]
    for _, row in all_df.iterrows():
        id_to_seq[str(row[id_col])] = row[seq_col]

    # 保存预测结果
    predictions = []
    for i, idx in enumerate(all_test_indices):
        predictions.append({
            'peptide_id': peptide_ids[idx],
            'sequence': id_to_seq.get(peptide_ids[idx], ""),
            'true_label': all_y_true[i],
            'predicted_label': all_y_pred[i],
            'probability': all_y_proba[i]
        })

    # 保存为CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    # 返回交叉验证平均指标
    return {
        'accuracy': avg_metrics['accuracy']['mean'] if 'accuracy' in avg_metrics else None,
        'precision': avg_metrics['precision']['mean'] if 'precision' in avg_metrics else None,
        'recall': avg_metrics['recall']['mean'] if 'recall' in avg_metrics else None,  # SN
        'specificity': avg_metrics['specificity']['mean'] if 'specificity' in avg_metrics else None,  # SP
        'f1': avg_metrics['f1']['mean'] if 'f1' in avg_metrics else None,
        'mcc': avg_metrics['mcc']['mean'] if 'mcc' in avg_metrics else None,
        'loss': avg_metrics['loss']['mean'] if 'loss' in avg_metrics else None,
        'auc': avg_metrics['auc']['mean'] if 'auc' in avg_metrics else None
    }


def main():

    """
    主函数，用于运行LSTM分类器训练和评估流程
    """
    # 设置默认文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_train_feature_file = os.path.join(current_dir,
                                              "processed_structures/peptidePrecursor_train_structures_peptide_features.pkl")
    default_test_feature_file = os.path.join(current_dir,
                                             "processed_structures/peptidePrecursor_test_structures_peptide_features.pkl")
    default_train_label_file = os.path.join(current_dir, "peptidePrecursor_dataset_training_id.csv")
    default_test_label_file = os.path.join(current_dir, "peptidePrecursor_dataset_testing_id.csv")
    default_output_dir = os.path.join(current_dir, "lstm_projection_fusion_results")

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="LSTM分类器带投影融合特征")
    parser.add_argument('--train_feature_file', type=str, default=default_train_feature_file,
                        help='训练集特征文件路径(.pkl)')
    parser.add_argument('--test_feature_file', type=str, default=default_test_feature_file,
                        help='测试集特征文件路径(.pkl)')
    parser.add_argument('--train_label_file', type=str, default=default_train_label_file,
                        help='训练集标签文件路径(.csv)')
    parser.add_argument('--test_label_file', type=str, default=default_test_label_file,
                        help='测试集标签文件路径(.csv)')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='结果输出目录')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='LSTM隐藏层维度')
    parser.add_argument('--projection_dim', type=int, default=512,
                        help='投影层输出维度，这个值影响特征降维程度，较大的值可保留更多特征信息')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比率')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--no_esm', action='store_true',
                        help='禁用ESM-2特征提取 (默认启用ESM特征)')
    parser.add_argument('--no_structure', action='store_true',
                        help='禁用结构特征 (默认启用结构特征)')
    parser.add_argument('--use_raw_sequence', action='store_true',
                        help='直接使用氨基酸序列数据供LSTM处理 (默认禁用)')
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D',
                        choices=['esm2_t33_650M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D'],
                        help='ESM-2模型类型')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子数')
    parser.add_argument('--gpu', type=str, default="cuda:1",
                        help='指定使用的GPU设备，如"cuda:0"或"cuda:2"，默认使用cuda:1')

    args = parser.parse_args()

    # 设置随机种子数以确保可重复性
    # set_all_seeds(args.seed)

    # 设置随机种子数以确保可重复性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 默认启用ESM特征和结构特征，除非使用--no_esm或--no_structure参数
    use_esm = not args.no_esm
    use_structure = not args.no_structure

    if args.no_structure:
        logger.info("禁用结构特征，将仅使用ESM特征（如果启用）或直接使用氨基酸序列")

    # 检查是否请求使用ESM，但模块不可用
    if use_esm and not ESM_AVAILABLE:
        logger.error("请求使用ESM-2特征，但ESM模块未安装。请使用pip install 'fair-esm'安装")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置计算设备
    if args.gpu is not None and torch.cuda.is_available():
        try:
            device = torch.device(args.gpu)
            logger.info(f"指定使用GPU设备: {args.gpu}")
        except Exception as e:
            logger.warning(f"无法使用指定的GPU设备 {args.gpu}: {e}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"使用设备: {device}")

    # 加载特征和标签
    X_orig, X_esm, y, peptide_ids, all_df, feature_dims = load_features_and_labels(
        args.test_feature_file, args.train_feature_file, args.train_label_file, args.test_label_file,
        use_esm=use_esm, use_structure=use_structure, use_raw_sequence=args.use_raw_sequence,
        esm_model=args.esm_model, projection_dim=args.projection_dim
    )

    # 初始化日志文件
    log_file = os.path.join(args.output_dir, 'lstm_classifier.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 记录实验配置信息
    logger.info("=" * 50)
    logger.info("实验配置:")
    logger.info("=" * 50)
    logger.info(f"使用结构特征: {use_structure}")
    logger.info(f"使用ESM特征: {use_esm}")
    logger.info(f"直接使用序列数据: {args.use_raw_sequence}")
    logger.info(f"特征维度信息: {feature_dims}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 50)

    # 如果使用embedding模式，记录额外信息
    if feature_dims.get('use_embedding', False):
        logger.info("使用嵌入层处理氨基酸序列:")
        logger.info(f"词汇表大小: {feature_dims.get('vocab_size', 21)}")
        logger.info(f"最大序列长度: {feature_dims.get('max_seq_length', 0)}")
        logger.info(f"嵌入维度: {feature_dims.get('embedding_dim', 64)}")

    # 原始特征标准化（如果不是序列数据）
    if not feature_dims.get('use_embedding', False):
        scaler_orig = StandardScaler()
        X_orig_scaled = scaler_orig.fit_transform(X_orig)
        logger.info(f"原始特征标准化后大小: {X_orig_scaled.shape}")

        # ESM特征标准化（如果有）
        if X_esm is not None:
            scaler_esm = StandardScaler()
            X_esm_scaled = scaler_esm.fit_transform(X_esm)
            logger.info(f"ESM特征标准化后大小: {X_esm_scaled.shape}")
        else:
            X_esm_scaled = None
    else:
        # 对于序列数据，不进行标准化
        X_orig_scaled = X_orig
        X_esm_scaled = X_esm
        logger.info("使用序列数据，跳过标准化步骤")

    # 增加GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
        memory_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
        logger.info(f"GPU内存状态 - 已分配: {memory_allocated:.2f} MB, 已保留: {memory_reserved:.2f} MB")

    # 记录模型结构信息
    logger.info("=" * 50)
    logger.info("LSTM模型配置:")
    logger.info("=" * 50)
    if feature_dims.get('use_embedding', False):
        logger.info(f"输入: 氨基酸序列 (vocab_size={feature_dims.get('vocab_size', 21)})")
        logger.info(f"嵌入维度: {feature_dims.get('embedding_dim', 64)}")
    elif X_esm is not None:
        logger.info(f"- 原始特征维度: {X_orig.shape[1]}")
        logger.info(f"- ESM特征维度: {X_esm.shape[1]}")
        logger.info(f"- 投影维度: {feature_dims.get('projection_dim', 512)}")
        logger.info(f"- 融合方法: 投影后拼接")
    else:
        logger.info(f"- 输入维度: {X_orig.shape[1]}")
    logger.info(f"- 隐藏层维度: {args.hidden_dim}")
    logger.info(f"- LSTM层数: {args.num_layers}")
    logger.info(f"- Dropout比率: {args.dropout}")
    logger.info("=" * 50)

    # 使用交叉验证训练和评估模型
    logger.info("开始交叉验证评估...")
    metrics = evaluate_model(
        X_orig_scaled, X_esm_scaled, y, peptide_ids, all_df, feature_dims,
        device, args.output_dir, args.batch_size
    )

    # 输出最终结果
    logger.info("=" * 50)
    logger.info("最终模型评估结果:")
    logger.info("=" * 50)
    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"精确率: {metrics['precision']:.4f}")
    logger.info(f"灵敏度(SN): {metrics['recall']:.4f}")
    logger.info(f"特异性(SP): {metrics['specificity']:.4f}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info(f"MCC: {metrics['mcc']:.4f}")
    logger.info(f"平均损失值: {metrics['loss']:.4f}")
    if metrics['auc'] is not None:
        logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info("=" * 50)

    logger.info(f"处理完成，结果保存在: {args.output_dir}")
    logger.info(f"最佳模型已保存在: {os.path.join(args.output_dir, 'models')}")



if __name__ == "__main__":
    main()
