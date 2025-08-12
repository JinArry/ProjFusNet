#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDB结构处理脚本 (Biopython版本)

该脚本用于处理一个目录下的所有PDB文件，提取结构特征，
并将其保存为二进制格式，以便于模型训练过程中更快地加载。
使用Biopython库提供更可靠的PDB文件解析和序列提取。

作者：JinjinLi
日期：2025-04-14
"""

import os
import sys
import argparse
import logging
import traceback
import pickle
import csv
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import *
from Bio.PDB.DSSP import DSSP, dssp_dict_from_pdb_file
from Bio.PDB.vectors import calc_dihedral
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from concurrent.futures import ProcessPoolExecutor
import freesasa
from scipy.spatial import distance_matrix, distance
from pathlib import Path
import argparse
import traceback
import warnings
import json
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from Bio import PDB
from Bio.PDB import PPBuilder
import glob
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist  # 添加cdist导入
import networkx as nx
import community.community_louvain as community_louvain
import scipy.stats
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

# 在文件开头的导入部分后添加警告过滤
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB")


# 检查DSSP是否可用
def check_dssp():
    try:
        from Bio.PDB.DSSP import dssp_dict_from_pdb_file
        return True
    except ImportError:
        logger.error("DSSP未安装或无法导入。请按照以下步骤安装DSSP：")
        logger.error("1. 安装Homebrew（如果尚未安装）：")
        logger.error(
            "   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        logger.error("2. 使用Homebrew安装DSSP：")
        logger.error("   brew install dssp")
        logger.error("3. 验证安装：")
        logger.error("   mkdssp --version")
        return False


# 在程序开始时检查DSSP
if not check_dssp():
    sys.exit(1)

# 添加必要的库
try:
    from Bio import PDB
    from Bio.PDB import PPBuilder

    BIOPYTHON_AVAILABLE = True
    logging.info("成功导入Biopython库")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.error("未能导入Biopython库，请安装: pip install biopython")
    sys.exit(1)

# 当前脚本中已包含序列提取功能，不需要外部模块
logging.info("使用内置的序列提取方法")

# 定义三字母氨基酸到单字母码的映射
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'XLE': 'J',
    'XAA': 'X'
}


# 内置序列提取函数
def extract_full_sequence(structure):
    """
    从结构中提取完整的氨基酸序列

    Args:
        structure: Biopython解析的PDB结构

    Returns:
        sequence: 完整的氨基酸序列字符串
    """
    sequence = ""

    # 遍历所有模型和链
    for model in structure:
        for chain in model:
            chain_seq = ""
            # 直接从残基中提取
            for residue in chain:
                if PDB.is_aa(residue):
                    res_name = residue.resname
                    aa_code = THREE_TO_ONE.get(res_name, 'X')
                    chain_seq += aa_code

            sequence += chain_seq

    return sequence


# 设置日志
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdb_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 增强重要信息的可见性
def log_important(message):
    """Log important information in a more visible way"""
    separator = "=" * 50
    logger.info(separator)
    logger.info(message)
    logger.info(separator)
    # 同时输出到控制台，以确保可见性
    print(f"\n{separator}\n{message}\n{separator}\n")


# 结构特征维度
# 由于我们添加了更多特征，调整预设维度以适应实际特征维度(83+)
# 定义结构特征维度 - 更新为包含新增特征
STRUCTURE_FEATURE_DIM = 128  # 增加维度以包含新特征128

# 要包含在CSV文件中的特征名称
FEATURE_NAMES = [
    'peptide_id', 'sequence', 'length',
    'avg_hydrophobicity', 'net_charge', 'avg_b_factor',
    'helix_percent', 'sheet_percent', 'coil_percent',
    'ss_H', 'ss_B', 'ss_E', 'ss_G', 'ss_I', 'ss_T', 'ss_S', 'ss_C',
    'avg_exposure', 'max_exposure', 'min_exposure',
    'avg_distance_to_center', 'max_local_density',
    'avg_phi', 'avg_psi', 'phi_psi_entropy',
    'avg_sasa', 'polar_sasa_ratio', 'hydrophobic_sasa_ratio',
    'avg_contacts', 'long_range_contacts', 'contact_order', 'contact_density',
    'charged_residues_density', 'charged_patch_count',
    'pocket_count', 'avg_pocket_volume', 'max_pocket_surface_area',
    'ptm_site_count', 'avg_stability_score',
    'solvent_accessibility', 'radius_of_gyration',
    'freq_A', 'freq_C', 'freq_D', 'freq_E', 'freq_F', 'freq_G', 'freq_H',
    'freq_I', 'freq_K', 'freq_L', 'freq_M', 'freq_N', 'freq_P', 'freq_Q',
    'freq_R', 'freq_S', 'freq_T', 'freq_V', 'freq_W', 'freq_Y',
    'plddt_mean', 'plddt_std', 'plddt_max', 'plddt_min',
    'max_ca_distance', 'surface_residue_ratio',
    'low_confidence_ratio', 'protein_volume', 'max_pocket_surface_area',
    'high_confidence_ratio',
    # 新增特征
    'surface_roughness', 'surface_charge_distribution', 'surface_hydrophobicity_distribution',
    'domain_count', 'inter_domain_contacts', 'inter_domain_distance',
    'backbone_hbond_count', 'sidechain_hbond_count', 'total_hbond_count',
    'chi1_entropy', 'chi2_entropy', 'chi3_entropy', 'chi4_entropy',
    'surface_polarity', 'core_polarity', 'surface_hydrophobicity', 'core_hydrophobicity',
    'contact_network_density', 'contact_clustering_coefficient', 'contact_centrality',
    'contact_modularity', 'contact_degree_distribution', 'contact_network_diameter',
    'surface_polarity', 'core_polarity', 'surface_hydrophobicity', 'core_hydrophobicity',
    'domain_size_avg', 'domain_size_std', 'domain_compactness', 'domain_connectivity',
    'domain_hierarchy', 'domain_interface_area', 'domain_interface_residues',
    'domain_interface_hydrophobicity', 'domain_interface_polarity', 'domain_stability',
    'domain_mobility',
    'hbond_count', 'hbond_density', 'hbond_energy', 'hbond_network_density',
    'hbond_clustering_coefficient', 'hbond_centrality', 'hbond_modularity',
    'hbond_degree_distribution', 'hbond_network_diameter', 'hbond_network_radius',
    'domain_mobility',
    'surface_residue_ratio', 'surface_polarity', 'surface_hydrophobicity', 'surface_charge',
    'surface_roughness', 'surface_curvature', 'surface_compactness', 'surface_elongation',
    'surface_connectivity', 'surface_hierarchy', 'surface_stability', 'surface_mobility',
    'surface_interface', 'surface_interface_residues', 'surface_interface_hydrophobicity',
    'surface_interface_polarity', 'surface_interface_area', 'surface_interface_compactness',
    'surface_interface_elongation', 'surface_interface_roughness', 'surface_interface_charge',
    'surface_interface_accessibility', 'surface_interface_connectivity', 'surface_interface_hierarchy',
    'surface_interface_stability', 'surface_interface_mobility'
]

# 氨基酸三字母到一字母的映射
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    # 非标准氨基酸映射为X
    'SEC': 'X', 'PYL': 'X', 'UNK': 'X'
}

# AlphaFold兼容的氨基酸编码映射 (0-20)
# 0-19为标准氨基酸，20为未知氨基酸
AA_TO_INDEX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20  # 未知氨基酸
}

# 氨基酸物理化学特征
# 疏水性映射 (Kyte & Doolittle疏水性指数)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0  # 未知氨基酸
}

# 电荷映射 (pH 7.4)
CHARGE = {
    'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': -1.0, 'C': 0.0,
    'E': -1.0, 'Q': 0.0, 'G': 0.0, 'H': 0.1, 'I': 0.0,
    'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
    'X': 0.0  # 未知氨基酸
}

# 侧链体积 ((立方埃))
SIDE_CHAIN_VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'E': 138.4, 'Q': 143.8, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
    'X': 138.4  # 未知氨基酸，使用平均值
}

# 等电点
# 氨基酸的等电点，表示其在中性pH条件下的电荷平衡点
ISOELECTRIC_POINT = {
    'A': 6.0, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07,
    'E': 3.22, 'Q': 5.65, 'G': 5.97, 'H': 7.59, 'I': 6.02,
    'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.30,
    'S': 5.68, 'T': 5.60, 'W': 5.89, 'Y': 5.66, 'V': 5.96,
    'X': 6.0  # 未知氨基酸，使用中性值
}

# 氨基酸保存性得分 (Karlin & Brocchieri 1996)
# 这些分数表示氨基酸在进化中的保存程度，越高表示越保存
CONSERVATION_SCORE = {
    'A': 0.7, 'R': 0.5, 'N': 0.3, 'D': 0.3, 'C': 0.9,
    'E': 0.4, 'Q': 0.4, 'G': 0.8, 'H': 0.4, 'I': 0.7,
    'L': 0.8, 'K': 0.4, 'M': 0.6, 'F': 0.7, 'P': 0.6,
    'S': 0.5, 'T': 0.5, 'W': 0.8, 'Y': 0.7, 'V': 0.8,
    'X': 0.5  # 未知氨基酸，使用中性值
}

# 氨基酸组成特征
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'


def aa_composition(sequence):
    length = len(sequence)
    comp = {}
    for aa in AA_LIST:
        comp[f'freq_{aa}'] = sequence.count(aa) / length if length > 0 else 0
    return comp


# DSSP 8分类统计
DSSP_CODES = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']


def dssp_8class_stats(structure):
    from Bio.PDB.DSSP import DSSP
    import tempfile
    import os
    from Bio.PDB.PDBIO import PDBIO
    stats = {f'ss_{code}': 0.0 for code in DSSP_CODES}
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            tmp_pdb = tmp.name
            io = PDBIO()
            io.set_structure(structure)
            io.save(tmp_pdb)
        dssp = DSSP(structure[0], tmp_pdb, dssp='mkdssp')
        total = len(dssp)
        if total > 0:
            for key in dssp.keys():
                ss = dssp[key][2]
                if ss not in DSSP_CODES:
                    ss = '-'
                stats[f'ss_{ss}'] += 1
            for code in DSSP_CODES:
                stats[f'ss_{code}'] /= total
    except Exception as e:
        pass
    finally:
        if 'tmp_pdb' in locals() and os.path.exists(tmp_pdb):
            os.unlink(tmp_pdb)
    return stats


# pLDDT分布统计
import numpy as np


def plddt_stats(structure):
    plddt_scores = []
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res):
                    for atom in res:
                        if atom.get_id() == 'CA':
                            plddt_scores.append(atom.get_bfactor())
    if plddt_scores:
        arr = np.array(plddt_scores)
        return {
            'plddt_mean': float(np.mean(arr)),
            'plddt_std': float(np.std(arr)),
            'plddt_max': float(np.max(arr)),
            'plddt_min': float(np.min(arr)),
        }
    else:
        return {'plddt_mean': 0.0, 'plddt_std': 0.0, 'plddt_max': 0.0, 'plddt_min': 0.0}


def extract_pdb_id(filename):
    """
    从PDB文件名中提取序列ID
    规则：去掉最后一个下划线及其后面的内容
    例如：CCG4_NEUCR_1982292.pdb -> CCG4_NEUCR
    """
    try:
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        if '_' in name_without_ext:
            sequence_id = '_'.join(name_without_ext.split('_')[:-1])
            return sequence_id
        return name_without_ext
    except Exception as e:
        logger.error(f"从文件名提取ID时出错: {filename}, 错误: {e}")
        return None


def calculate_dihedral_angles(structure):
    """
    计算蛋白质主链的二面角(phi, psi)

    Args:
        structure: Biopython解析的PDB结构

    Returns:
        phi_angles: phi角度列表
        psi_angles: psi角度列表
    """
    phi_angles = []
    psi_angles = []

    try:
        # 获取第一个模型
        model = structure[0]

        # 遍历所有链
        for chain in model:
            # 创建残基列表
            residues = []
            for res in chain:
                if PDB.is_aa(res) and res.id[0] == ' ':  # 只考虑标准氨基酸
                    residues.append(res)

            # 确保残基按序列顺序排序
            residues.sort(key=lambda x: x.id[1])

            # 计算phi和psi角度
            for i in range(len(residues)):
                res = residues[i]

                # 计算phi角 (需要前一个残基)
                if i > 0:
                    prev = residues[i - 1]
                    # 检查是否有必要的原子
                    if ('C' in prev) and ('N' in res) and ('CA' in res) and ('C' in res):
                        try:
                            # 计算二面角 C(i-1)-N(i)-CA(i)-C(i)
                            phi = calc_dihedral(prev['C'].get_vector(),
                                                res['N'].get_vector(),
                                                res['CA'].get_vector(),
                                                res['C'].get_vector())
                            phi_angles.append(math.degrees(phi))
                        except Exception as e:
                            logging.debug(f"计算phi角时出错: {e}")

                # 计算psi角 (需要后一个残基)
                if i < len(residues) - 1:
                    next_res = residues[i + 1]
                    # 检查是否有必要的原子
                    if ('N' in res) and ('CA' in res) and ('C' in res) and ('N' in next_res):
                        try:
                            # 计算二面角 N(i)-CA(i)-C(i)-N(i+1)
                            psi = calc_dihedral(res['N'].get_vector(),
                                                res['CA'].get_vector(),
                                                res['C'].get_vector(),
                                                next_res['N'].get_vector())
                            psi_angles.append(math.degrees(psi))
                        except Exception as e:
                            logging.debug(f"计算psi角时出错: {e}")

    except Exception as e:
        logging.error(f"计算二面角时出错: {e}")

    return phi_angles, psi_angles


def calculate_dihedral_features(phi_angles, psi_angles):
    """
    计算二面角特征
    """
    features = {
        'phi_angle_mean': 0.0,
        'phi_angle_std': 0.0,
        'psi_angle_mean': 0.0,
        'psi_angle_std': 0.0,
        'phi_psi_correlation': 0.0,
        'ramachandran_entropy': 0.0
    }

    try:
        if len(phi_angles) > 0 and len(psi_angles) > 0:
            # 计算基本统计量
            features['phi_angle_mean'] = float(np.mean(phi_angles))
            features['phi_angle_std'] = float(np.std(phi_angles))
            features['psi_angle_mean'] = float(np.mean(psi_angles))
            features['psi_angle_std'] = float(np.std(psi_angles))

            # 计算相关性
            if len(phi_angles) > 1:
                correlation = np.corrcoef(phi_angles, psi_angles)[0, 1]
                features['phi_psi_correlation'] = float(correlation if not np.isnan(correlation) else 0.0)

            # 计算Ramachandran图熵
            hist, _, _ = np.histogram2d(phi_angles, psi_angles, bins=36, range=[[-180, 180], [-180, 180]])
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
                valid_bins = hist > 0
                if np.any(valid_bins):
                    features['ramachandran_entropy'] = float(-np.sum(hist[valid_bins] * np.log2(hist[valid_bins])))

    except Exception as e:
        logger.error(f"计算二面角特征时出错: {e}")
        logger.error(traceback.format_exc())

    return features


def calculate_sasa_features(structure):
    """
    计算溶剂可及表面积特征
    """
    features = {
        'avg_sasa': 0.0,
        'max_sasa': 0.0,
        'min_sasa': 0.0,
        'sasa_std': 0.0,
        'polar_sasa_ratio': 0.0,
        'hydrophobic_sasa_ratio': 0.0,
        'surface_residue_ratio': 0.0,
        'buried_residue_ratio': 0.0,
        'sasa_per_residue': 0.0,
        'sasa_distribution': 0.0
    }

    try:
        # 计算SASA
        import freesasa
        import tempfile
        from Bio.PDB.PDBIO import PDBIO

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            sasa_structure = freesasa.Structure(temp_file_path)
            result = freesasa.calc(sasa_structure)
            residue_areas = result.residueAreas()

            # 收集所有残基的SASA值
            sasa_values = []
            polar_sasa = 0.0
            hydrophobic_sasa = 0.0
            surface_residues = 0
            buried_residues = 0
            total_residues = 0

            for model in structure:
                for chain in model:
                    for res in chain:
                        if PDB.is_aa(res) and res.id[0] == ' ':
                            total_residues += 1
                            res_name = res.get_resname()
                            chain_id = chain.id
                            res_id = res.id[1]

                            # 尝试获取SASA值
                            possible_keys = [
                                f"{chain_id}:{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id}",
                                f"{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id} ",
                                f"{res_name} {res_id}"
                            ]

                            sasa = 0.0
                            for key in possible_keys:
                                if key in residue_areas:
                                    sasa = residue_areas[key][0]
                                    break

                            sasa_values.append(sasa)

                            # 计算极性和疏水性SASA
                            if res_name in ['ARG', 'LYS', 'HIS', 'ASP', 'GLU', 'ASN', 'GLN', 'SER', 'THR', 'TYR']:
                                polar_sasa += sasa
                            if res_name in ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL']:
                                hydrophobic_sasa += sasa

                            # 统计表面和埋藏残基
                            if sasa > 5.0:  # 表面残基阈值
                                surface_residues += 1
                            else:
                                buried_residues += 1

            if sasa_values:
                sasa_values = np.array(sasa_values)
                features['avg_sasa'] = float(np.mean(sasa_values))
                features['max_sasa'] = float(np.max(sasa_values))
                features['min_sasa'] = float(np.min(sasa_values))
                features['sasa_std'] = float(np.std(sasa_values))
                features['sasa_per_residue'] = float(np.sum(sasa_values) / total_residues)
                features['sasa_distribution'] = float(np.var(sasa_values))

                # 计算比率
                total_sasa = np.sum(sasa_values)
                if total_sasa > 0:
                    features['polar_sasa_ratio'] = float(polar_sasa / total_sasa)
                    features['hydrophobic_sasa_ratio'] = float(hydrophobic_sasa / total_sasa)

                if total_residues > 0:
                    features['surface_residue_ratio'] = float(surface_residues / total_residues)
                    features['buried_residue_ratio'] = float(buried_residues / total_residues)

            # 添加调试信息
            logger.info(f"Total residues: {total_residues}")
            logger.info(f"Surface residues: {surface_residues}")
            logger.info(f"Buried residues: {buried_residues}")
            logger.info(f"Average SASA: {features['avg_sasa']}")
            logger.info(f"Polar SASA ratio: {features['polar_sasa_ratio']}")
            logger.info(f"Hydrophobic SASA ratio: {features['hydrophobic_sasa_ratio']}")

        except Exception as e:
            logger.error(f"计算SASA时出错: {e}")

    except Exception as e:
        logger.error(f"计算SASA特征时出错: {e}")

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return features


def get_residue_sasa_dict(structure):
    """
    返回每个残基的SASA值，键为 (chain_id, res_id)
    """
    import freesasa
    import tempfile
    from Bio.PDB.PDBIO import PDBIO
    sasa_dict = {}
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_file.name)
            temp_file_path = temp_file.name
        sasa_structure = freesasa.Structure(temp_file_path)
        result = freesasa.calc(sasa_structure)
        residue_areas = result.residueAreas()
        for model in structure:
            for chain in model:
                for res in chain:
                    if PDB.is_aa(res) and res.id[0] == ' ':
                        res_name = res.get_resname()
                        chain_id = chain.id
                        res_id = res.id[1]
                        # 多种可能的key
                        possible_keys = [
                            f"{chain_id}:{res_name}{res_id}",
                            f"{chain_id}:{res_name} {res_id}",
                            f"{res_name}{res_id}",
                            f"{chain_id}:{res_name} {res_id} ",
                            f"{res_name} {res_id}"
                        ]
                        sasa = 0.0
                        for key in possible_keys:
                            if key in residue_areas:
                                sasa = residue_areas[key][0]
                                break
                        sasa_dict[(chain_id, res_id)] = sasa
    except Exception as e:
        logger.error(f"获取残基SASA时出错: {e}")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return sasa_dict


def calculate_contact_features(structure):
    """
    计算接触特征
    """
    features = {
        'avg_contacts': 0.0,
        'long_range_contacts': 0,
        'contact_order': 0.0,
        'contact_density': 0.0
    }

    try:
        # 获取所有CA原子
        ca_atoms = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if PDB.is_aa(res) and res.id[0] == ' ' and 'CA' in res:
                        ca_atoms.append((res, res['CA'].get_coord()))

        if not ca_atoms:
            return features

        # 计算接触
        contacts = []
        long_range_contacts = 0
        total_contacts = 0

        for i, (res1, coord1) in enumerate(ca_atoms):
            res1_contacts = 0
            for j, (res2, coord2) in enumerate(ca_atoms):
                if i != j:
                    dist = np.linalg.norm(coord1 - coord2)
                    if dist < 8.0:  # 接触距离阈值
                        res1_contacts += 1
                        total_contacts += 1
                        if abs(res1.get_id()[1] - res2.get_id()[1]) > 12:  # 长程接触
                            long_range_contacts += 1
            contacts.append(res1_contacts)

        if contacts:
            features['avg_contacts'] = float(np.mean(contacts))
            features['long_range_contacts'] = long_range_contacts
            features['contact_order'] = float(long_range_contacts) / total_contacts if total_contacts > 0 else 0.0
            features['contact_density'] = float(total_contacts) / (len(ca_atoms) * (len(ca_atoms) - 1))

    except Exception as e:
        logger.error(f"计算接触特征时出错: {e}")

    return features


def calculate_pocket_features(structure):
    """
    计算口袋特征，包括形状、深度分布和连通性等特征
    """
    features = {
        'pocket_count': 0,
        'pocket_volume_avg': 0.0,
        'pocket_volume_std': 0.0,
        'pocket_surface_area_avg': 0.0,
        'pocket_surface_area_std': 0.0,
        'pocket_depth_avg': 0.0,
        'pocket_depth_std': 0.0,
        'pocket_compactness_avg': 0.0,  # 新增：口袋紧凑度
        'pocket_compactness_std': 0.0,
        'pocket_connectivity_avg': 0.0,  # 新增：口袋连通性
        'pocket_connectivity_std': 0.0,
        'pocket_aspect_ratio_avg': 0.0,  # 新增：口袋长宽比
        'pocket_aspect_ratio_std': 0.0,
        'pocket_depth_distribution': 0.0,  # 新增：口袋深度分布
        'pocket_volume_distribution': 0.0,  # 新增：口袋体积分布
        'pocket_surface_roughness_avg': 0.0,  # 新增：口袋表面粗糙度
        'pocket_surface_roughness_std': 0.0
    }
    try:
        ca_coords = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if PDB.is_aa(res) and 'CA' in res:
                        ca_coords.append(res['CA'].get_coord())

        if len(ca_coords) < 10:
            logger.warning("结构中CA原子数量不足")
            return features

        ca_coords = np.array(ca_coords)
        min_coords = np.min(ca_coords, axis=0)
        max_coords = np.max(ca_coords, axis=0)

        # 创建更精细的网格点
        grid_size = 2.0  # 临时调大步长以加速
        x_grid = np.arange(min_coords[0], max_coords[0] + grid_size, grid_size)
        y_grid = np.arange(min_coords[1], max_coords[1] + grid_size, grid_size)
        z_grid = np.arange(min_coords[2], max_coords[2] + grid_size, grid_size)

        # 创建KD树用于快速最近邻搜索
        kdtree = cKDTree(ca_coords)

        # 生成网格点
        grid_points = np.array(np.meshgrid(x_grid, y_grid, z_grid)).T.reshape(-1, 3)

        # 计算每个网格点到最近CA原子的距离
        distances, _ = kdtree.query(grid_points, k=1)

        # 使用更合理的距离范围筛选口袋点
        pocket_points = grid_points[(distances > 1.2) & (distances < 3.5)]  # 调整距离范围

        if len(pocket_points) > 0:
            # 使用更合适的DBSCAN参数进行聚类
            clustering = DBSCAN(eps=1.8, min_samples=3).fit(pocket_points)  # 调整eps
            labels = clustering.labels_

            # 统计有效口袋数量
            valid_pockets = 0
            pocket_volumes = []
            pocket_surface_areas = []
            pocket_depths = []
            pocket_compactnesses = []
            pocket_connectivities = []
            pocket_aspect_ratios = []
            pocket_surface_roughnesses = []

            for pocket_id in range(len(set(labels)) - (1 if -1 in labels else 0)):
                pocket_coords = pocket_points[labels == pocket_id]
                if len(pocket_coords) > 4:  # 至少需要4个点才能形成凸包
                    try:
                        hull = ConvexHull(pocket_coords)
                        volume = hull.volume
                        if volume > 8.0:  # 降低体积阈值以检测更多口袋
                            valid_pockets += 1
                            pocket_volumes.append(volume)
                            pocket_surface_areas.append(hull.area)

                            # 计算口袋深度
                            centroid = np.mean(pocket_coords, axis=0)
                            depths = np.linalg.norm(pocket_coords - centroid, axis=1)
                            pocket_depths.append(np.mean(depths))

                            # 计算口袋紧凑度 (36πV²)^(1/3)/A
                            compactness = (36 * np.pi * volume ** 2) ** (1 / 3) / hull.area
                            pocket_compactnesses.append(compactness)

                            # 计算口袋连通性
                            connectivity = 0
                            for i in range(len(pocket_coords)):
                                for j in range(i + 1, len(pocket_coords)):
                                    if np.linalg.norm(pocket_coords[i] - pocket_coords[j]) < 2.0:
                                        connectivity += 1
                            if len(pocket_coords) > 1:
                                connectivity = 2 * connectivity / (len(pocket_coords) * (len(pocket_coords) - 1))
                                pocket_connectivities.append(connectivity)

                            # 计算口袋长宽比
                            try:
                                # 使用PCA计算主成分
                                centered_coords = pocket_coords - centroid
                                cov_matrix = np.cov(centered_coords.T)
                                eigenvalues = np.linalg.eigvals(cov_matrix)
                                aspect_ratio = np.sqrt(np.max(eigenvalues) / np.min(eigenvalues))
                                pocket_aspect_ratios.append(aspect_ratio)
                            except:
                                pocket_aspect_ratios.append(1.0)

                            # 计算口袋表面粗糙度
                            try:
                                # 使用局部曲率估计表面粗糙度
                                local_curvatures = []
                                for point in pocket_coords:
                                    # 找到最近的k个点
                                    distances, indices = kdtree.query(point, k=5)
                                    if len(indices) > 1:
                                        # 计算局部曲率
                                        local_points = pocket_coords[indices]
                                        local_centroid = np.mean(local_points, axis=0)
                                        local_radius = np.mean(np.linalg.norm(local_points - local_centroid, axis=1))
                                        local_curvature = 1.0 / local_radius if local_radius > 0 else 0
                                        local_curvatures.append(local_curvature)
                                if local_curvatures:
                                    pocket_surface_roughnesses.append(np.mean(local_curvatures))
                            except:
                                pocket_surface_roughnesses.append(0.0)
                    except:
                        continue

            features['pocket_count'] = valid_pockets

            if valid_pockets > 0:
                if pocket_volumes:
                    features['pocket_volume_avg'] = float(np.mean(pocket_volumes))
                    features['pocket_volume_std'] = float(np.std(pocket_volumes))
                    features['pocket_volume_distribution'] = float(np.var(pocket_volumes))

                if pocket_surface_areas:
                    features['pocket_surface_area_avg'] = float(np.mean(pocket_surface_areas))
                    features['pocket_surface_area_std'] = float(np.std(pocket_surface_areas))

                if pocket_depths:
                    features['pocket_depth_avg'] = float(np.mean(pocket_depths))
                    features['pocket_depth_std'] = float(np.std(pocket_depths))
                    features['pocket_depth_distribution'] = float(np.var(pocket_depths))

                if pocket_compactnesses:
                    features['pocket_compactness_avg'] = float(np.mean(pocket_compactnesses))
                    features['pocket_compactness_std'] = float(np.std(pocket_compactnesses))

                if pocket_connectivities:
                    features['pocket_connectivity_avg'] = float(np.mean(pocket_connectivities))
                    features['pocket_connectivity_std'] = float(np.std(pocket_connectivities))

                if pocket_aspect_ratios:
                    features['pocket_aspect_ratio_avg'] = float(np.mean(pocket_aspect_ratios))
                    features['pocket_aspect_ratio_std'] = float(np.std(pocket_aspect_ratios))

                if pocket_surface_roughnesses:
                    features['pocket_surface_roughness_avg'] = float(np.mean(pocket_surface_roughnesses))
                    features['pocket_surface_roughness_std'] = float(np.std(pocket_surface_roughnesses))

                logger.info(f"检测到 {valid_pockets} 个有效口袋")
                logger.info(f"平均口袋体积: {features['pocket_volume_avg']:.2f}")
                logger.info(f"平均口袋表面积: {features['pocket_surface_area_avg']:.2f}")
                logger.info(f"平均口袋紧凑度: {features['pocket_compactness_avg']:.2f}")
                logger.info(f"平均口袋连通性: {features['pocket_connectivity_avg']:.2f}")

    except Exception as e:
        logger.error(f"计算口袋特征时出错: {e}")
        logger.error(traceback.format_exc())

    return features


def calculate_secondary_structure_features(structure):
    """
    计算二级结构特征
    """
    features = {
        'helix_percent': 0.0,
        'sheet_percent': 0.0,
        'coil_percent': 0.0,
        'turn_percent': 0.0,
        'bridge_percent': 0.0,
        'bend_percent': 0.0,
        'secondary_structure_transitions': 0.0,
        'secondary_structure_entropy': 0.0
    }
    try:
        import tempfile
        from Bio.PDB.PDBIO import PDBIO
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            dssp = DSSP(structure[0], temp_file_path, dssp='mkdssp')
            ss_types = {'H': 0, 'G': 0, 'I': 0, 'E': 0, 'B': 0, 'T': 0, 'S': 0, ' ': 0, '-': 0}  # 添加'-'类型
            ss_sequence = []

            for k in dssp.keys():
                ss = dssp[k][2]  # 获取二级结构类型
                if ss not in ss_types:  # 处理未知的二级结构类型
                    ss = ' '  # 将未知类型视为无规卷曲
                ss_types[ss] += 1
                ss_sequence.append(ss)

            total = sum(ss_types.values())
            if total > 0:
                # 计算各类二级结构的百分比
                features['helix_percent'] = float((ss_types['H'] + ss_types['G'] + ss_types['I']) / total)
                features['sheet_percent'] = float((ss_types['E'] + ss_types['B']) / total)
                features['coil_percent'] = float(
                    (ss_types['T'] + ss_types['S'] + ss_types[' '] + ss_types['-']) / total)
                features['turn_percent'] = float(ss_types['T'] / total)
                features['bridge_percent'] = float(ss_types['B'] / total)
                features['bend_percent'] = float(ss_types['S'] / total)

                # 计算二级结构转变
                if len(ss_sequence) > 1:
                    transitions = np.zeros((9, 9))  # 9种二级结构类型（包括'-'）
                    ss_to_idx = {'H': 0, 'G': 1, 'I': 2, 'E': 3, 'B': 4, 'T': 5, 'S': 6, ' ': 7, '-': 8}

                    for i in range(len(ss_sequence) - 1):
                        curr_ss = ss_sequence[i]
                        next_ss = ss_sequence[i + 1]
                        if curr_ss in ss_to_idx and next_ss in ss_to_idx:
                            transitions[ss_to_idx[curr_ss]][ss_to_idx[next_ss]] += 1

                    # 计算转移概率矩阵
                    row_sums = np.sum(transitions, axis=1)
                    # 避免除以零，将零行替换为均匀分布
                    zero_rows = row_sums == 0
                    if np.any(zero_rows):
                        transitions[zero_rows] = 1.0 / transitions.shape[1]
                        row_sums[zero_rows] = 1.0
                    transitions = transitions / row_sums[:, np.newaxis]

                    # 计算转变熵
                    valid_transitions = transitions[transitions > 0]
                    if len(valid_transitions) > 0:
                        features['secondary_structure_transitions'] = float(
                            -np.sum(valid_transitions * np.log2(valid_transitions)))

                    # 计算二级结构熵
                    ss_counts = np.array(list(ss_types.values()))
                    ss_sum = np.sum(ss_counts)
                    if ss_sum > 0:
                        ss_probs = ss_counts / ss_sum
                        features['secondary_structure_entropy'] = float(-np.sum(ss_probs * np.log2(ss_probs + 1e-10)))

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        logger.error(f"计算二级结构时出错: {e}")
        logger.error(traceback.format_exc())
    return features


def extract_avg_plddt(structure):
    """
    从结构中提取pLDDT均值（AlphaFold结构通常将pLDDT存储在B-factor列）
    """
    plddt_scores = []
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res):
                    for atom in res:
                        # 只统计主链CA原子的B-factor作为pLDDT
                        if atom.get_id() == 'CA':
                            plddt_scores.append(atom.get_bfactor())
    if plddt_scores:
        return sum(plddt_scores) / len(plddt_scores)
    else:
        return None


def max_ca_distance(structure):
    ca_coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res) and 'CA' in res:
                    ca_coords.append(res['CA'].get_coord())
    if len(ca_coords) < 2:
        return 0.0
    return float(np.max(pdist(np.array(ca_coords))))


def surface_residue_ratio(structure, sasa_threshold=10.0):  # 进一步降低SASA阈值
    import freesasa
    import tempfile
    from Bio.PDB.PDBIO import PDBIO
    surface_count = 0
    total = 0
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_file.name)
            temp_file_path = temp_file.name
        sasa_structure = freesasa.Structure(temp_file_path)
        result = freesasa.calc(sasa_structure)

        # 获取所有残基的SASA值
        residue_areas = result.residueAreas()
        logger.info(f"SASA结果键: {list(residue_areas.keys())[:5]}")  # 打印前5个键的格式

        # 遍历所有残基
        for model in structure:
            for chain in model:
                for res in chain:
                    if PDB.is_aa(res) and res.id[0] == ' ':
                        total += 1
                        res_name = res.get_resname()
                        chain_id = chain.id
                        res_id = res.id[1]
                        # 尝试多种可能的键格式
                        possible_keys = [
                            f"{chain_id}:{res_name}{res_id}",
                            f"{chain_id}:{res_name} {res_id}",
                            f"{chain_id}:{res_name}{res_id} ",
                            f"{res_name}{res_id}",
                            f"{res_name} {res_id}"
                        ]

                        sasa = 0.0
                        for key in possible_keys:
                            if key in residue_areas:
                                sasa = residue_areas[key][0]
                                break

                        if sasa > sasa_threshold:
                            surface_count += 1

        logger.info(
            f"表面残基比例计算: 总残基数={total}, 表面残基数={surface_count}, 比例={surface_count / total if total > 0 else 0}")
    except Exception as e:
        logger.error(f"计算表面残基比例时出错: {e}")
        return 0.0
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return surface_count / total if total > 0 else 0.0


def low_confidence_ratio(structure, threshold=70):
    scores = []
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res):
                    for atom in res:
                        if atom.get_id() == 'CA':
                            scores.append(atom.get_bfactor())
    if not scores:
        return 0.0
    scores = np.array(scores)
    return float(np.sum(scores < threshold) / len(scores))


# 蛋白质体积（用freesasa）
def protein_volume(structure):
    import freesasa
    import tempfile
    from Bio.PDB.PDBIO import PDBIO
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_file.name)
            temp_file_path = temp_file.name
        sasa_structure = freesasa.Structure(temp_file_path)
        result = freesasa.calc(sasa_structure)
        # 使用totalArea()方法获取表面积，然后估算体积
        surface_area = float(result.totalArea())
        # 使用经验公式估算体积：V ≈ A^(3/2) / 6π
        volume = (surface_area ** 1.5) / (6 * math.pi)
        logger.info(f"蛋白质体积计算: 表面积={surface_area}, 估算体积={volume}")
        return volume
    except Exception as e:
        logger.error(f"计算蛋白质体积时出错: {e}")
        return 0.0
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# 最大口袋表面积（基于pocket_features的聚类点凸包面积）
def max_pocket_surface_area(structure):
    from sklearn.cluster import DBSCAN
    from scipy.spatial import ConvexHull
    from scipy.spatial import cKDTree
    import numpy as np
    ca_coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res) and 'CA' in res:
                    ca_coords.append(res['CA'].get_coord())
    if len(ca_coords) < 10:
        return 0.0
    ca_coords = np.array(ca_coords)
    min_coords = np.min(ca_coords, axis=0)
    max_coords = np.max(ca_coords, axis=0)
    grid_size = 1.5
    x_grid = np.arange(min_coords[0], max_coords[0] + grid_size, grid_size)
    y_grid = np.arange(min_coords[1], max_coords[1] + grid_size, grid_size)
    z_grid = np.arange(min_coords[2], max_coords[2] + grid_size, grid_size)
    kdtree = cKDTree(ca_coords)
    pocket_points = []
    for x in x_grid:
        for y in y_grid:
            for z in z_grid:
                point = np.array([x, y, z])
                min_dist, _ = kdtree.query(point)
                if 2.0 <= min_dist <= 5.0:
                    pocket_points.append(point)
    if not pocket_points:
        return 0.0
    pocket_points = np.array(pocket_points)
    dbscan = DBSCAN(eps=2.5, min_samples=10)
    labels = dbscan.fit_predict(pocket_points)
    unique_labels = np.unique(labels[labels != -1])
    max_area = 0.0
    for label in unique_labels:
        cluster = pocket_points[labels == label]
        if len(cluster) >= 4:
            try:
                hull = ConvexHull(cluster)
                area = hull.area
                if area > max_area:
                    max_area = area
            except Exception:
                continue
    return float(max_area)


# pLDDT高置信度比例（pLDDT > 90）
def high_confidence_ratio(structure, threshold=90):
    scores = []
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res):
                    for atom in res:
                        if atom.get_id() == 'CA':
                            scores.append(atom.get_bfactor())
    if not scores:
        return 0.0
    scores = np.array(scores)
    return float(np.sum(scores > threshold) / len(scores))


def calc_seq_physicochem(sequence, structure):
    # 只保留标准氨基酸
    std_aa = set('ACDEFGHIKLMNPQRSTVWY')
    clean_seq = ''.join([aa for aa in sequence if aa in std_aa])
    # 平均疏水性和净电荷
    try:
        pa = ProteinAnalysis(clean_seq)
        avg_hydro = pa.gravy() if clean_seq else 0.0
        net_charge = pa.charge_at_pH(7.0) if clean_seq else 0.0
    except Exception as e:
        avg_hydro = 0.0
        net_charge = 0.0
    # 平均B因子
    b_factors = []
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res):
                    for atom in res:
                        if atom.get_id() == 'CA':
                            b_factors.append(atom.get_bfactor())
    avg_b = float(np.mean(b_factors)) if b_factors else 0.0
    return {
        'avg_hydrophobicity': avg_hydro,
        'net_charge': net_charge,
        'avg_b_factor': avg_b
    }


def calculate_charged_residues_density(structure):
    """
    计算蛋白质带电残基密度（带电残基数/体积）
    """
    charged_residues = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS'}
    charged_count = 0
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res) and res.id[0] == ' ':
                    if res.get_resname() in charged_residues:
                        charged_count += 1
    volume = protein_volume(structure)
    density = charged_count / volume if volume > 0 else 0.0
    logger.info(f"带电残基密度计算: 带电残基数={charged_count}, 体积={volume}, 密度={density}")
    return density


def calculate_charged_patch_count(structure, sasa_threshold=5.0):
    """
    计算带电残基斑块数量
    """
    try:
        sasa_dict = get_residue_sasa_dict(structure)
        if not sasa_dict:
            return 0
        # 获取表面残基
        surface_residues = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if PDB.is_aa(res) and res.id[0] == ' ':
                        sasa = sasa_dict.get((chain.id, res.id[1]), 0.0)
                        if sasa > sasa_threshold:
                            surface_residues.append(res)
        if not surface_residues:
            return 0
        # 识别带电残基
        charged_residues = []
        for res in surface_residues:
            res_name = res.get_resname()
            if res_name in ['ARG', 'LYS', 'HIS', 'ASP', 'GLU']:
                charged_residues.append(res)
        if not charged_residues:
            return 0
        coords = np.array([res['CA'].get_coord() for res in charged_residues if 'CA' in res])
        if len(coords) == 0:
            return 0
        clustering = DBSCAN(eps=3.0, min_samples=1).fit(coords)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        logger.info(f"Charged patches: {n_clusters}")
        return n_clusters
    except Exception as e:
        logger.error(f"计算带电斑块数量时出错: {e}")
        return 0


def calculate_chi_angles(structure):
    """
    计算侧链二面角特征
    """
    features = {
        'chi1_angles_avg': 0.0,
        'chi1_angles_std': 0.0,
        'chi2_angles_avg': 0.0,
        'chi2_angles_std': 0.0,
        'chi3_angles_avg': 0.0,
        'chi3_angles_std': 0.0,
        'chi4_angles_avg': 0.0,
        'chi4_angles_std': 0.0
    }
    try:
        chi1_angles = []
        chi2_angles = []
        chi3_angles = []
        chi4_angles = []

        for model in structure:
            for chain in model:
                for res in chain:
                    if not PDB.is_aa(res):
                        continue
                    try:
                        # 计算chi1角
                        if 'N' in res and 'CA' in res and 'CB' in res and 'CG' in res:
                            chi1 = PDB.calc_dihedral(res['N'].get_vector(),
                                                     res['CA'].get_vector(),
                                                     res['CB'].get_vector(),
                                                     res['CG'].get_vector())
                            chi1_angles.append(chi1)

                        # 计算chi2角
                        if 'CA' in res and 'CB' in res and 'CG' in res and 'CD' in res:
                            chi2 = PDB.calc_dihedral(res['CA'].get_vector(),
                                                     res['CB'].get_vector(),
                                                     res['CG'].get_vector(),
                                                     res['CD'].get_vector())
                            chi2_angles.append(chi2)

                        # 计算chi3角
                        if 'CB' in res and 'CG' in res and 'CD' in res and 'CE' in res:
                            chi3 = PDB.calc_dihedral(res['CB'].get_vector(),
                                                     res['CG'].get_vector(),
                                                     res['CD'].get_vector(),
                                                     res['CE'].get_vector())
                            chi3_angles.append(chi3)

                        # 计算chi4角
                        if 'CG' in res and 'CD' in res and 'CE' in res and 'NZ' in res:
                            chi4 = PDB.calc_dihedral(res['CG'].get_vector(),
                                                     res['CD'].get_vector(),
                                                     res['CE'].get_vector(),
                                                     res['NZ'].get_vector())
                            chi4_angles.append(chi4)
                    except:
                        continue

        # 使用len()检查列表是否为空，而不是直接对numpy数组进行布尔判断
        if len(chi1_angles) > 0:
            features['chi1_angles_avg'] = float(np.mean(chi1_angles))
            features['chi1_angles_std'] = float(np.std(chi1_angles))

        if len(chi2_angles) > 0:
            features['chi2_angles_avg'] = float(np.mean(chi2_angles))
            features['chi2_angles_std'] = float(np.std(chi2_angles))

        if len(chi3_angles) > 0:
            features['chi3_angles_avg'] = float(np.mean(chi3_angles))
            features['chi3_angles_std'] = float(np.std(chi3_angles))

        if len(chi4_angles) > 0:
            features['chi4_angles_avg'] = float(np.mean(chi4_angles))
            features['chi4_angles_std'] = float(np.std(chi4_angles))

    except Exception as e:
        logger.error(f"计算侧链二面角特征时出错: {e}")
        logger.error(traceback.format_exc())
    return features


def calculate_surface_features(structure):
    """
    计算表面特征
    """
    features = {
        'surface_area': 0.0,
        'surface_roughness': 0.0,
        'surface_curvature': 0.0,
        'surface_hydrophobicity': 0.0,
        'surface_charge': 0.0,
        'surface_polarity': 0.0,
        'surface_aromaticity': 0.0,
        'surface_flexibility': 0.0,
        'surface_conservation': 0.0,
        'surface_accessibility': 0.0
    }

    try:
        import freesasa
        import tempfile
        from Bio.PDB.PDBIO import PDBIO

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            sasa_structure = freesasa.Structure(temp_file_path)
            result = freesasa.calc(sasa_structure)  # 移除算法参数
            total_sasa = result.totalArea()
            features['surface_area'] = float(total_sasa)

            # 计算表面粗糙度
            if total_sasa > 0:
                residue_areas = result.residueAreas()
                area_values = []
                for model in structure:
                    for chain in model:
                        for res in chain:
                            if PDB.is_aa(res) and res.id[0] == ' ':
                                res_name = res.get_resname()
                                chain_id = chain.id
                                res_id = res.id[1]
                                possible_keys = [
                                    f"{chain_id}:{res_name}{res_id}",
                                    f"{chain_id}:{res_name} {res_id}",
                                    f"{res_name}{res_id}",
                                    f"{chain_id}:{res_name} {res_id} ",
                                    f"{res_name} {res_id}"
                                ]
                                for key in possible_keys:
                                    if key in residue_areas:
                                        area_values.append(residue_areas[key][0])
                                        break

                if area_values:
                    area_values = np.array(area_values)
                    features['surface_roughness'] = float(np.std(area_values) / np.mean(area_values))

            # 计算表面疏水性
            hydrophobic_residues = {'ALA', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL'}
            hydrophobic_sasa = 0.0
            for model in structure:
                for chain in model:
                    for res in chain:
                        if PDB.is_aa(res) and res.id[0] == ' ':
                            res_name = res.get_resname()
                            chain_id = chain.id
                            res_id = res.id[1]
                            possible_keys = [
                                f"{chain_id}:{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id}",
                                f"{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id} ",
                                f"{res_name} {res_id}"
                            ]
                            for key in possible_keys:
                                if key in residue_areas:
                                    if res_name in hydrophobic_residues:
                                        hydrophobic_sasa += residue_areas[key][0]
                                    break

            if total_sasa > 0:
                features['surface_hydrophobicity'] = float(hydrophobic_sasa / total_sasa)

            # 计算表面电荷
            charged_residues = {'ARG', 'LYS', 'HIS', 'ASP', 'GLU'}
            charged_sasa = 0.0
            for model in structure:
                for chain in model:
                    for res in chain:
                        if PDB.is_aa(res) and res.id[0] == ' ':
                            res_name = res.get_resname()
                            chain_id = chain.id
                            res_id = res.id[1]
                            possible_keys = [
                                f"{chain_id}:{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id}",
                                f"{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id} ",
                                f"{res_name} {res_id}"
                            ]
                            for key in possible_keys:
                                if key in residue_areas:
                                    if res_name in charged_residues:
                                        charged_sasa += residue_areas[key][0]
                                    break

            if total_sasa > 0:
                features['surface_charge'] = float(charged_sasa / total_sasa)

            # 计算表面极性
            polar_residues = {'ARG', 'LYS', 'HIS', 'ASP', 'GLU', 'ASN', 'GLN', 'SER', 'THR', 'TYR'}
            polar_sasa = 0.0
            for model in structure:
                for chain in model:
                    for res in chain:
                        if PDB.is_aa(res) and res.id[0] == ' ':
                            res_name = res.get_resname()
                            chain_id = chain.id
                            res_id = res.id[1]
                            possible_keys = [
                                f"{chain_id}:{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id}",
                                f"{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id} ",
                                f"{res_name} {res_id}"
                            ]
                            for key in possible_keys:
                                if key in residue_areas:
                                    if res_name in polar_residues:
                                        polar_sasa += residue_areas[key][0]
                                    break

            if total_sasa > 0:
                features['surface_polarity'] = float(polar_sasa / total_sasa)

            # 计算表面芳香性
            aromatic_residues = {'PHE', 'TRP', 'TYR'}
            aromatic_sasa = 0.0
            for model in structure:
                for chain in model:
                    for res in chain:
                        if PDB.is_aa(res) and res.id[0] == ' ':
                            res_name = res.get_resname()
                            chain_id = chain.id
                            res_id = res.id[1]
                            possible_keys = [
                                f"{chain_id}:{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id}",
                                f"{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id} ",
                                f"{res_name} {res_id}"
                            ]
                            for key in possible_keys:
                                if key in residue_areas:
                                    if res_name in aromatic_residues:
                                        aromatic_sasa += residue_areas[key][0]
                                    break

            if total_sasa > 0:
                features['surface_aromaticity'] = float(aromatic_sasa / total_sasa)

            # 计算表面可及性
            accessible_residues = 0
            total_residues = 0
            for model in structure:
                for chain in model:
                    for res in chain:
                        if PDB.is_aa(res) and res.id[0] == ' ':
                            total_residues += 1
                            res_name = res.get_resname()
                            chain_id = chain.id
                            res_id = res.id[1]
                            possible_keys = [
                                f"{chain_id}:{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id}",
                                f"{res_name}{res_id}",
                                f"{chain_id}:{res_name} {res_id} ",
                                f"{res_name} {res_id}"
                            ]
                            for key in possible_keys:
                                if key in residue_areas:
                                    if residue_areas[key][0] > 5.0:  # 表面残基阈值
                                        accessible_residues += 1
                                    break

            if total_residues > 0:
                features['surface_accessibility'] = float(accessible_residues / total_residues)

            # 添加调试信息
            logger.info(f"Surface area: {features['surface_area']}")
            logger.info(f"Surface roughness: {features['surface_roughness']}")
            logger.info(f"Surface hydrophobicity: {features['surface_hydrophobicity']}")
            logger.info(f"Surface charge: {features['surface_charge']}")
            logger.info(f"Surface polarity: {features['surface_polarity']}")
            logger.info(f"Surface aromaticity: {features['surface_aromaticity']}")
            logger.info(f"Surface accessibility: {features['surface_accessibility']}")

        except Exception as e:
            logger.error(f"计算表面特征时出错: {e}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"计算表面特征时出错: {e}")
        logger.error(traceback.format_exc())

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return features


def calculate_domain_features(structure):
    """
    计算结构域特征
    """
    features = {
        'domain_count': 0,
        'domain_size_avg': 0.0,
        'domain_size_std': 0.0,
        'domain_compactness': 0.0,
        'domain_connectivity': 0.0,
        'domain_hierarchy': 0.0,
        'domain_interface_area': 0.0,
        'domain_interface_residues': 0.0,
        'domain_interface_hydrophobicity': 0.0,
        'domain_interface_polarity': 0.0,
        'domain_stability': 0.0,
        'domain_mobility': 0.0
    }
    try:
        # 收集所有CA原子坐标
        ca_coords = []
        ca_residues = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if PDB.is_aa(res) and 'CA' in res:
                        ca_coords.append(res['CA'].get_coord())
                        ca_residues.append((chain.id, res.id))

        if len(ca_coords) < 10:
            return features

        ca_coords = np.array(ca_coords)

        # 使用DBSCAN进行域识别
        clustering = DBSCAN(eps=8.0, min_samples=5).fit(ca_coords)
        labels = clustering.labels_

        # 统计域数量
        domain_count = len(set(labels)) - (1 if -1 in labels else 0)
        features['domain_count'] = domain_count

        if domain_count > 0:
            # 计算每个域的特征
            domain_sizes = []
            domain_compactnesses = []
            domain_connectivities = []

            for domain_id in range(domain_count):
                domain_coords = ca_coords[labels == domain_id]
                domain_residues = [ca_residues[i] for i, label in enumerate(labels) if label == domain_id]

                if len(domain_coords) > 4:  # 至少需要4个点才能形成凸包
                    # 计算域大小
                    domain_sizes.append(len(domain_coords))

                    # 计算域紧凑度
                    try:
                        hull = ConvexHull(domain_coords)
                        volume = hull.volume
                        surface_area = hull.area
                        if surface_area > 0:
                            compactness = (36 * np.pi * volume ** 2) ** (1 / 3) / surface_area
                            domain_compactnesses.append(compactness)
                    except:
                        continue

                    # 计算域连接性
                    try:
                        # 计算域内残基间的接触
                        contacts = 0
                        for i in range(len(domain_coords)):
                            for j in range(i + 1, len(domain_coords)):
                                if np.linalg.norm(domain_coords[i] - domain_coords[j]) < 8.0:
                                    contacts += 1
                        if len(domain_coords) > 1:
                            connectivity = 2 * contacts / (len(domain_coords) * (len(domain_coords) - 1))
                            domain_connectivities.append(connectivity)
                    except:
                        continue

            # 更新特征
            if domain_sizes:
                features['domain_size_avg'] = float(np.mean(domain_sizes))
                features['domain_size_std'] = float(np.std(domain_sizes))

            if domain_compactnesses:
                features['domain_compactness'] = float(np.mean(domain_compactnesses))

            if domain_connectivities:
                features['domain_connectivity'] = float(np.mean(domain_connectivities))

            # 计算域层次性
            if len(domain_sizes) > 1:
                features['domain_hierarchy'] = float(np.std(domain_sizes) / np.mean(domain_sizes))

            # 计算域界面特征
            if domain_count > 1:
                interface_areas = []
                interface_residues = []
                interface_hydrophobicities = []
                interface_polarities = []

                for i in range(domain_count):
                    for j in range(i + 1, domain_count):
                        domain_i_coords = ca_coords[labels == i]
                        domain_j_coords = ca_coords[labels == j]

                        # 计算域间接触
                        interface_count = 0
                        interface_hydrophobicity = 0.0
                        interface_polarity = 0.0

                        for coord_i in domain_i_coords:
                            for coord_j in domain_j_coords:
                                if np.linalg.norm(coord_i - coord_j) < 8.0:
                                    interface_count += 1
                                    # 获取残基的物理化学性质
                                    res_i = ca_residues[np.where(labels == i)[0][
                                        np.argmin(np.linalg.norm(domain_i_coords - coord_i, axis=1))]]
                                    res_j = ca_residues[np.where(labels == j)[0][
                                        np.argmin(np.linalg.norm(domain_j_coords - coord_j, axis=1))]]

                                    res_i_name = structure[0][res_i[0]][res_i[1]].get_resname()
                                    res_j_name = structure[0][res_j[0]][res_j[1]].get_resname()

                                    if res_i_name in HYDROPHOBICITY:
                                        interface_hydrophobicity += HYDROPHOBICITY[res_i_name]
                                        interface_polarity += POLARITY[res_i_name]
                                    if res_j_name in HYDROPHOBICITY:
                                        interface_hydrophobicity += HYDROPHOBICITY[res_j_name]
                                        interface_polarity += POLARITY[res_j_name]

                        if interface_count > 0:
                            interface_areas.append(interface_count * 20.0)  # 假设每个接触贡献20 Å²
                            interface_residues.append(interface_count)
                            interface_hydrophobicities.append(interface_hydrophobicity / interface_count)
                            interface_polarities.append(interface_polarity / interface_count)

                if interface_areas:
                    features['domain_interface_area'] = float(np.mean(interface_areas))
                    features['domain_interface_residues'] = float(np.mean(interface_residues))
                    features['domain_interface_hydrophobicity'] = float(np.mean(interface_hydrophobicities))
                    features['domain_interface_polarity'] = float(np.mean(interface_polarities))

            # 计算域稳定性（基于B因子）
            try:
                b_factors = []
                for model in structure:
                    for chain in model:
                        for res in chain:
                            if PDB.is_aa(res) and 'CA' in res:
                                b_factors.append(res['CA'].get_bfactor())
                if b_factors:
                    features['domain_stability'] = float(1.0 / (np.mean(b_factors) + 1e-10))
            except:
                pass

            # 计算域流动性（基于B因子的变异）
            try:
                if b_factors:
                    features['domain_mobility'] = float(np.std(b_factors) / (np.mean(b_factors) + 1e-10))
            except:
                pass

    except Exception as e:
        logger.error(f"计算结构域特征时出错: {e}")
        logger.error(traceback.format_exc())
    return features


def calculate_hbond_features(structure):
    """
    计算氢键特征
    """
    features = {
        'hbond_count': 0,
        'hbond_density': 0.0,
        'hbond_energy_avg': 0.0,
        'hbond_energy_std': 0.0,
        'hbond_distance_avg': 0.0,
        'hbond_distance_std': 0.0,
        'hbond_angle_avg': 0.0,
        'hbond_angle_std': 0.0
    }
    try:
        # 使用临时文件进行DSSP分析
        import tempfile
        from Bio.PDB.PDBIO import PDBIO
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            dssp = DSSP(structure[0], temp_file_path, dssp='mkdssp')
            hbonds = []
            for model in structure:
                for chain in model:
                    for res1 in chain:
                        if not PDB.is_aa(res1):
                            continue
                        for res2 in chain:
                            if not PDB.is_aa(res2):
                                continue
                            if res1.id[1] >= res2.id[1]:
                                continue
                            try:
                                # 计算氢键
                                hbond = PDB.HBond(res1, res2)
                                if hbond:
                                    hbonds.append(hbond)
                            except:
                                continue

            if hbonds:
                features['hbond_count'] = len(hbonds)
                features['hbond_density'] = len(hbonds) / len(list(structure.get_residues()))
                features['hbond_energy_avg'] = float(np.mean([hb.energy for hb in hbonds]))
                features['hbond_energy_std'] = float(np.std([hb.energy for hb in hbonds]))
                features['hbond_distance_avg'] = float(np.mean([hb.distance for hb in hbonds]))
                features['hbond_distance_std'] = float(np.std([hb.distance for hb in hbonds]))
                features['hbond_angle_avg'] = float(np.mean([hb.angle for hb in hbonds]))
                features['hbond_angle_std'] = float(np.std([hb.angle for hb in hbonds]))

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        logger.error(f"计算氢键特征时出错: {e}")
        logger.error(traceback.format_exc())
    return features


def process_pdb_file(args):
    """
    处理单个PDB文件，提取结构特征

    Args:
        args: 包含PDB文件路径和标签的元组

    Returns:
        peptide_id: 蛋白质ID
        features: 特征字典
        metadata: 元数据字典
    """
    pdb_file, label = args
    try:
        # 从文件名中提取ID，只保留蛋白质ID部分（去掉后面的数字）
        filename = os.path.basename(pdb_file)
        pdb_id = filename.split('.')[0]  # 去掉.pdb后缀
        peptide_id = '_'.join(pdb_id.split('_')[:-1])  # 去掉最后的数字部分

        # 解析PDB文件
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)

        # 获取序列
        sequence = extract_full_sequence(structure)

        # 初始化特征字典
        features = {
            'peptide_id': peptide_id,  # 使用处理后的peptide_id
            'sequence': sequence,
            'length': len(sequence)
        }

        # 1. 二级结构特征
        ss_features = calculate_secondary_structure_features(structure)
        features.update(ss_features)

        # 2. 溶剂可及性特征
        sasa_features = calculate_sasa_features(structure)
        features.update(sasa_features)

        # 3. 接触图特征
        contact_features = calculate_contact_features(structure)
        features.update(contact_features)

        # 4. 口袋特征
        pocket_features = calculate_pocket_features(structure)
        features.update(pocket_features)

        # 5. 氨基酸组成特征
        aa_features = aa_composition(sequence)
        features.update(aa_features)

        # 6. 主链二面角特征
        phi_psi_features = calculate_dihedral_features(
            calculate_dihedral_angles(structure)[0],
            calculate_dihedral_angles(structure)[1]
        )
        features.update(phi_psi_features)

        # 7. 侧链二面角特征
        chi_features = calculate_chi_angles(structure)
        features.update(chi_features)

        # 8. 物理化学特征
        physicochemical_features = calc_seq_physicochem(sequence, structure)
        features.update(physicochemical_features)

        # 9. 带电残基特征
        charged_features = {
            'charged_residues_density': calculate_charged_residues_density(structure),
            'charged_patch_count': calculate_charged_patch_count(structure)
        }
        features.update(charged_features)

        # 10. 结构质量特征
        quality_features = {
            'avg_plddt': extract_avg_plddt(structure),
            'high_confidence_ratio': high_confidence_ratio(structure),
            'low_confidence_ratio': low_confidence_ratio(structure)
        }
        features.update(quality_features)

        # 11. 结构特征
        structure_features = {
            'max_ca_distance': max_ca_distance(structure),
            'surface_residue_ratio': surface_residue_ratio(structure),
            'protein_volume': protein_volume(structure),
            'max_pocket_surface_area': max_pocket_surface_area(structure)
        }
        features.update(structure_features)

        # 记录处理结果
        logger.info(f"成功处理文件: {pdb_file}")
        logger.info(f"序列长度: {len(sequence)}")
        logger.info(f"二级结构: 螺旋 {ss_features['helix_percent']:.2%}, 折叠 {ss_features['sheet_percent']:.2%}")
        logger.info(f"口袋数量: {pocket_features['pocket_count']}")

        # 返回结果
        return peptide_id, features, {
            'sequence': sequence,
            'label': label,
            'file_path': pdb_file
        }

    except Exception as e:
        logger.error(f"处理文件 {pdb_file} 时出错: {e}")
        logger.error(traceback.format_exc())
        return None, None, None


def process_pdb_directories(pdb_dirs, label_files, output_dir, num_workers=4, max_files=None):
    """
    处理多个目录下的所有PDB文件

    Args:
        pdb_dirs: PDB文件所在目录的列表
        label_files: 对应每个目录的标签文件路径列表
        output_dir: 输出特征目录
        num_workers: 并行处理的工作进程数
        max_files: 最多处理的PDB文件数（用于快速测试）

    Returns:
        成功处理的股肢ID列表和全部特征数据
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    all_pdb_files = []
    all_id_to_label = {}

    # 收集所有PDB文件和标签映射
    for pdb_dir, label_file in zip(pdb_dirs, label_files):
        # 加载标签映射
        id_to_label = {}
        if label_file:
            id_to_label = load_id_to_label_mapping(label_file)
            logger.info(f"从 {label_file} 中加载了 {len(id_to_label)} 个 ID-标签映射")
            all_id_to_label.update(id_to_label)

        # 获取PDB文件
        pdb_files = []
        for root, _, files in os.walk(pdb_dir):
            for file in files:
                if file.endswith('.pdb'):
                    pdb_files.append((os.path.join(root, file), id_to_label))
        # 新增：只取前max_files个
        if max_files is not None and len(pdb_files) > max_files:
            pdb_files = pdb_files[:max_files]

        logger.info(f"在 {pdb_dir} 中找到 {len(pdb_files)} 个PDB文件")
        all_pdb_files.extend(pdb_files)

    # 新增：全局只取前max_files个
    if max_files is not None and len(all_pdb_files) > max_files:
        all_pdb_files = all_pdb_files[:max_files]

    if not all_pdb_files:
        logger.warning("在所有目录中未找到PDB文件")
        return [], None, None

    # 创建CSV文件用于存储特征元数据
    csv_file = os.path.join(output_dir, "structure_features.csv")
    processed_ids = []
    all_features = {}
    all_metadata = {}

    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 准备参数，包含标签映射
        args_list = [(pdb_file, label) for pdb_file, label in all_pdb_files]
        # 使用tqdm显示进度
        results = list(tqdm(executor.map(process_pdb_file, args_list),
                            total=len(args_list),
                            desc="处理PDB文件"))

    # 收集成功处理的结果和失败的结果
    successful_results = [r for r in results if r and r[0] and r[1] is not False]
    failed_results = [r for r in results if r and r[0] and r[1] is False]
    failed_ids = [r[0] for r in failed_results if r and r[0]]

    logger.info(f"成功处理 {len(successful_results)} 个PDB文件")
    logger.info(f"处理失败 {len(failed_ids)} 个PDB文件")

    if not successful_results:
        logger.warning("没有成功处理的PDB文件")
        return [], None, None

    # 收集所有元数据和特征
    for peptide_id, features, metadata in successful_results:
        processed_ids.append(peptide_id)
        all_features[peptide_id] = features
        all_metadata[peptide_id] = metadata

    # 保存失败ID列表
    if failed_ids:
        failed_ids_file = os.path.join(output_dir, "failed_ids.txt")
        try:
            with open(failed_ids_file, 'w') as f:
                for peptide_id in failed_ids:
                    f.write(f"{peptide_id}\n")
            logger.info(f"失败ID列表已保存到 {failed_ids_file}")
        except Exception as e:
            logger.error(f"保存失败ID列表时出错: {e}")

    # 保存特征数据
    try:
        # 保存所有特征到PKL文件
        all_features_file = os.path.join(output_dir, "all_structure_features.pkl")
        with open(all_features_file, 'wb') as f:
            pickle.dump(all_features, f)
        logger.info(f"将 {len(all_features)} 个结构的特征数据保存到 {all_features_file}")

        # 保存特征到CSV文件
        if all_features:
            rows = []
            for peptide_id in processed_ids:
                if peptide_id in all_metadata and peptide_id in all_features:
                    # 合并所有特征
                    row = {**all_features[peptide_id]}  # 直接使用特征字典，因为它已经包含了peptide_id
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                # 确保列的顺序与FEATURE_NAMES一致
                ordered_columns = [col for col in FEATURE_NAMES if col != 'label' and col in df.columns]
                df = df[ordered_columns]
                df.to_csv(csv_file, index=False)
                logger.info(f"将 {len(rows)} 条全特征数据保存到 {csv_file}")

    except Exception as e:
        logger.error(f"保存特征数据时出错: {e}")
        logger.error(traceback.format_exc())

    return processed_ids, all_features, all_metadata


def load_id_to_label_mapping(csv_file):
    """
    从 CSV 文件中加载 ID 和标签的映射

    Args:
        csv_file: CSV 文件路径，包含 ID 和 label 列

    Returns:
        id_to_label: 从 ID 到标签的映射字典
    """
    id_to_label = {}
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file):
            logger.error(f"CSV 文件不存在: {csv_file}")
            return id_to_label

        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 检查必要的列是否存在
        required_columns = ['uniProtkbId', 'label']
        if not all(col in df.columns for col in required_columns):
            # 尝试其他可能的列名
            if 'peptide_id' in df.columns and 'label' in df.columns:
                id_column = 'peptide_id'
            elif 'id' in df.columns and 'label' in df.columns:
                id_column = 'id'
            else:
                logger.error(f"CSV 文件缺少必要的列: {csv_file}")
                return id_to_label
        else:
            id_column = 'uniProtkbId'

        # 创建ID到标签的映射
        for _, row in df.iterrows():
            id_to_label[row[id_column]] = row['label']

        logger.info(f"从 {csv_file} 中加载了 {len(id_to_label)} 个 ID-标签映射")
    except Exception as e:
        logger.error(f"加载 CSV 文件时出错: {csv_file}, 错误: {e}")
        logger.error(traceback.format_exc())

    return id_to_label


def main():
    """主函数"""
    # 获取当前脚本路径和相关目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"当前工作目录: {current_dir}")

    # 设置输入和输出目录
    pdb_dir = os.path.join(current_dir, 'Data/structure_local')
    output_dir = os.path.join(current_dir, 'Data/structure_features')
    logger.info(f"PDB文件目录: {pdb_dir}")
    logger.info(f"输出目录: {output_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test_output'), exist_ok=True)

    # 测试模式：处理特定序列
    test_mode = False  # 设置为True启用测试模式
    if test_mode:
        # 要测试的序列ID列表 - 使用实际存在的PDB文件
        test_sequences = [
            'CAH3_HUMAN',
            'WBPP_PSEAI'
        ]

        # 创建测试输出目录
        test_output_dir = os.path.join(output_dir, 'test_output')
        os.makedirs(test_output_dir, exist_ok=True)

        # 存储所有测试结果
        all_test_results = []
        all_features_dict = {}  # 存储所有特征
        total_amino_acids = 0  # 统计氨基酸总数
        failed_sequences = []  # 存储处理失败的序列

        # 处理每个测试序列
        for seq_id in test_sequences:
            logger.info(f"\n处理序列: {seq_id}")

            # 构建可能的PDB文件名模式
            pdb_file_pattern = os.path.join(pdb_dir, f"{seq_id}_*.pdb")
            matching_files = glob.glob(pdb_file_pattern)

            if not matching_files:
                logger.error(f"未找到PDB文件: {pdb_file_pattern}")
                failed_sequences.append((seq_id, "未找到PDB文件"))
                all_test_results.append({
                    'peptide_id': seq_id,
                    'sequence': '',
                    'length': 0,
                    'status': 'failed',
                    'error': '未找到PDB文件'
                })
                continue

            # 使用第一个匹配的文件
            pdb_file = matching_files[0]
            logger.info(f"开始处理文件: {pdb_file}")
            try:
                # 验证PDB文件
                if not os.path.exists(pdb_file):
                    raise FileNotFoundError(f"PDB文件不存在: {pdb_file}")

                if os.path.getsize(pdb_file) == 0:
                    raise ValueError(f"PDB文件为空: {pdb_file}")

                peptide_id, features, metadata = process_pdb_file((pdb_file, None))

                if features is not False and features is not None:
                    # 验证特征和元数据
                    if not isinstance(features, dict):
                        raise ValueError(f"特征数据格式错误: {type(features)}")
                    if not isinstance(metadata, dict):
                        raise ValueError(f"元数据格式错误: {type(metadata)}")

                    logger.info(f"成功处理文件: {peptide_id}")
                    logger.info(f"特征数量: {len(features)}")
                    logger.info(f"元数据: {metadata}")

                    # 将特征添加到字典中
                    all_features_dict[peptide_id] = features

                    # 更新氨基酸总数
                    if 'sequence' in metadata and metadata['sequence']:
                        total_amino_acids += len(metadata['sequence'])

                    # 收集结果
                    metadata['status'] = 'success'
                    all_test_results.append(metadata)
                else:
                    error_msg = "处理文件失败: 特征提取失败"
                    logger.error(error_msg)
                    failed_sequences.append((seq_id, error_msg))
                    all_test_results.append({
                        'peptide_id': seq_id,
                        'sequence': '',
                        'length': 0,
                        'status': 'failed',
                        'error': error_msg
                    })
            except Exception as e:
                error_msg = f"处理文件时出错: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                failed_sequences.append((seq_id, error_msg))
                all_test_results.append({
                    'peptide_id': seq_id,
                    'sequence': '',
                    'length': 0,
                    'status': 'failed',
                    'error': error_msg
                })

        # 保存失败序列信息
        if failed_sequences:
            failed_sequences_file = os.path.join(test_output_dir, "failed_sequences.txt")
            try:
                with open(failed_sequences_file, 'w') as f:
                    for seq_id, error in failed_sequences:
                        f.write(f"{seq_id}: {error}\n")
                logger.info(f"失败序列信息已保存到 {failed_sequences_file}")
            except Exception as e:
                logger.error(f"保存失败序列信息时出错: {e}")

        # 统计结构置信度（如avg_plddt）的平均值
        avg_plddts = []
        for features in all_features_dict.values():
            if 'avg_plddt' in features and features['avg_plddt'] is not None:
                avg_plddts.append(features['avg_plddt'])
        if avg_plddts:
            mean_plddt = sum(avg_plddts) / len(avg_plddts)
            std_plddt = np.std(avg_plddts) if len(avg_plddts) > 1 else 0
        else:
            mean_plddt = None
            std_plddt = None

        # 打印统计信息
        print("\n=== 特征提取统计信息 ===")
        print(f"总序列数量: {len(test_sequences)}")
        print(f"成功处理的序列数量: {len(all_features_dict)}")
        print(f"处理失败的序列数量: {len(failed_sequences)}")
        print(f"提取的氨基酸总数: {total_amino_acids}")
        print(f"平均序列长度: {total_amino_acids / len(all_features_dict) if all_features_dict else 0:.2f}")
        print(f"结构置信度（avg_plddt）均值: {mean_plddt if mean_plddt is not None else '无数据'}")
        print(f"结构置信度（avg_plddt）标准差: {std_plddt if std_plddt is not None else '无数据'}")
        print("======================\n")

        logger.info("\n=== 特征提取统计信息 ===")
        logger.info(f"总序列数量: {len(test_sequences)}")
        logger.info(f"成功处理的序列数量: {len(all_features_dict)}")
        logger.info(f"处理失败的序列数量: {len(failed_sequences)}")
        logger.info(f"提取的氨基酸总数: {total_amino_acids}")
        logger.info(f"平均序列长度: {total_amino_acids / len(all_features_dict) if all_features_dict else 0:.2f}")
        logger.info(f"结构置信度（avg_plddt）均值: {mean_plddt if mean_plddt is not None else '无数据'}")
        logger.info(f"结构置信度（avg_plddt）标准差: {std_plddt if std_plddt is not None else '无数据'}")
        logger.info("======================\n")

        # 保存所有测试结果到CSV
        test_metadata_file = os.path.join(test_output_dir, "test_results.csv")
        pd.DataFrame(all_test_results).to_csv(test_metadata_file, index=False)
        logger.info(f"所有测试结果已保存到: {test_metadata_file}")

        # 保存所有特征到一个.pkl文件
        all_features_file = os.path.join(test_output_dir, "all_features.pkl")
        with open(all_features_file, 'wb') as f:
            pickle.dump(all_features_dict, f)
        logger.info(f"所有特征已保存到: {all_features_file}")

        # 同时保存为CSV格式
        if all_features_dict:
            features_list = []
            for peptide_id, features in all_features_dict.items():
                features['peptide_id'] = peptide_id
                features_list.append(features)

            features_df = pd.DataFrame(features_list)
            # 确保列的顺序与FEATURE_NAMES一致
            ordered_columns = [col for col in FEATURE_NAMES if col != 'label' and col in features_df.columns]
            features_df = features_df[ordered_columns]
            features_df.to_csv(os.path.join(test_output_dir, "test_features.csv"), index=False)
            logger.info("所有特征已保存到 test_features.csv")

        return

    # 正常模式：处理所有文件
    processed_ids, all_features, all_metadata = process_pdb_directories(
        [pdb_dir],
        [None],
        output_dir,
        num_workers=6,
        # num_workers=4,
        # max_files=30  # 只处理前10个PDB文件用于快速测试
    )

    logger.info("PDB结构处理完成")
    logger.info(f"成功处理的结构数量: {len(processed_ids)}")

    # 如果成功处理了PDB文件并提取了特征，保存特征数据
    if processed_ids and all_features:
        try:
            # 保存特征到CSV文件
            csv_file = os.path.join(output_dir, "structure_features.csv")
            rows = []
            for peptide_id in processed_ids:
                if peptide_id in all_metadata and peptide_id in all_features:
                    # 合并所有特征
                    row = {**all_features[peptide_id]}  # 直接使用特征字典，因为它已经包含了peptide_id
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                # 确保列的顺序与FEATURE_NAMES一致
                ordered_columns = [col for col in FEATURE_NAMES if col != 'label' and col in df.columns]
                df = df[ordered_columns]
                df.to_csv(csv_file, index=False)
                logger.info(f"将 {len(rows)} 条全特征数据保存到 {csv_file}")

        except Exception as e:
            logger.error(f"保存特征数据时出错: {e}")
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    print('Start Running')
    main()
    print('Finished Running')