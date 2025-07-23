from Bio import PDB

# 读取PDB文件
pdb_file = "training_1_9355418.pdb"
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("protein", pdb_file)

# 提取蛋白质序列
for model in structure:
    for chain in model:
        sequence = []
        for residue in chain:
            if PDB.is_aa(residue):  # 检查是否为标准氨基酸
                sequence.append(PDB.Polypeptide.three_to_one(residue.resname))  # 三字母转一字母
        seq_str = ''.join(sequence)
        print(f"链ID: {chain.id}, 序列: {seq_str}")