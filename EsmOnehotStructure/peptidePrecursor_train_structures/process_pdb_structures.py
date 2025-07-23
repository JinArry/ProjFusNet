#从pdb文件中提取蛋白质序列的三级结构特征
import logging
import os
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdb_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_pdb_id(filename):
    """
        从PDB文件名中提取ID
        Args:
            filename: PDB文件名(如 testing_1000_2652437.pdb)
        Returns:
            peptide_id: 肽ID (如 testing_1000)
        """
    try:
    # 提取文件名
        basename = os.path.basename(filename)
        if basename.endswith(".pdb"):
            # 删除pdb后缀
            basename_no_ext = basename.replace(".pdb", "")
            parts = basename_no_ext.split("_")
            if len(parts) >= 2:
                peptide_id = f"{parts[0]}_{parts[1]}"
                return peptide_id
            else:
                logger.warn("文件名格式不符合预期 (testing_xxxx_yyyy.pdb")
            return basename_no_ext
    except:
        logger.error(f"从文件名提取ID失败：{filename},错误:{e}")
        return None

if __name__ == "__main__":
    extract_pdb_id()