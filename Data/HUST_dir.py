"""
HUST数据集数据文件目录
HUSTdata中每个文件中振动数据点约122000个数据点
"""
import os
from pathlib import Path

root_dir = r"/hy-tmp/Data/dataset/HUST"


# def get_filename(root_path):
#     file = os.listdir(path=root_path)
#     file_list = [os.path.join(root_path, f) for f in file]
#     if len(file_list) != 1:
#         print(f"There are {len(file_list)} in '{root_path}'. Please check your file!")
#         print('exit.')
#         exit()
#     return file_list[0]

def get_filename(root_path):
    """
    返回一个“确定的 Excel 文件路径”：
      - 若传入的是文件路径：直接返回该文件
      - 若传入的是目录：仅在目录内寻找 .xlsx/.xlsm（忽略 .npy / 临时文件），
        * 找不到 → 抛错
        * 找到多个 → 自动选择“修改时间最新”的一个（不再退出）
    """
    p = Path(root_path)

    # 已经是文件就直接用（兼容你现有的调用）
    if p.is_file():
        return str(p.resolve())

    if not p.is_dir():
        raise FileNotFoundError(f"Not found: {root_path}")

    # 只看 Excel，忽略 .npy、临时文件、目录等
    excels = [f for f in p.iterdir()
              if f.is_file()
              and f.suffix.lower() in ('.xlsx', '.xlsm')
              and not f.name.startswith('~$')]

    if not excels:
        raise FileNotFoundError(f"No .xlsx/.xlsm in '{root_path}'")

    if len(excels) > 1:
        pick = max(excels, key=lambda x: x.stat().st_mtime).resolve()
        print(f"[get_filename] Found {len(excels)} Excel files; pick latest: {pick.name}")
        return str(pick)

    return str(excels[0].resolve())

# H
H = [r'H/L1', r'H/L2', r'H/L3', r'H/L4']
H_L1 = get_filename(os.path.join(root_dir, H[0]))
H_L2 = get_filename(os.path.join(root_dir, H[1]))
H_L3 = get_filename(os.path.join(root_dir, H[2]))
H_L4 = get_filename(os.path.join(root_dir, H[3]))

# B
B_high_file = [r'high/B/L1', r'high/B/L2', r'high/B/L3', r'high/B/L4']
B_middle_file = [r'middle/B/L1', r'middle/B/L2', r'middle/B/L3', r'middle/B/L4']
B_high = [get_filename(os.path.join(root_dir, f)) for f in B_high_file]
B_middle = [get_filename(os.path.join(root_dir, f)) for f in B_middle_file]

# C
C_high_file = [r'high/C/L1', r'high/C/L2', r'high/C/L3', r'high/C/L4']
C_middle_file = [r'middle/C/L1', r'middle/C/L2', r'middle/C/L3', r'middle/C/L4']
C_high = [get_filename(os.path.join(root_dir, f)) for f in C_high_file]
C_middle = [get_filename(os.path.join(root_dir, f)) for f in C_middle_file]

# I
I_high_file = [r'high/I/L1', r'high/I/L2', r'high/I/L3', r'high/I/L4']
I_middle_file = [r'middle/I/L1', r'middle/I/L2', r'middle/I/L3', r'middle/I/L4']
I_high = [get_filename(os.path.join(root_dir, f)) for f in I_high_file]
I_middle = [get_filename(os.path.join(root_dir, f)) for f in I_middle_file]

# O
O_high_file = [r'high/O/L1', r'high/O/L2', r'high/O/L3', r'high/O/L4']
O_middle_file = [r'middle/O/L1', r'middle/O/L2', r'middle/O/L3', r'middle/O/L4']
O_high = [get_filename(os.path.join(root_dir, f)) for f in O_high_file]
O_middle = [get_filename(os.path.join(root_dir, f)) for f in O_middle_file]
#
hust_L1 = [H_L1, B_high[0], B_middle[0],  C_high[0], C_middle[0], I_high[0], I_middle[0], O_high[0], O_middle[0]]
hust_L2 = [H_L2, B_high[1], B_middle[1],  C_high[1], C_middle[1], I_high[1], I_middle[1], O_high[1], O_middle[1]]
hust_L3 = [H_L3, B_high[2], B_middle[2], C_high[2], C_middle[2], I_high[2],  I_middle[2], O_high[2],  O_middle[2]]
hust_L4 = [H_L4, B_high[3], B_middle[3],  C_high[3], C_middle[3], I_high[3],  I_middle[3], O_high[3],  O_middle[3]]


if __name__ == '__main__':
    pass
