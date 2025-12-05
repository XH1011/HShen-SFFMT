import numpy as np
import sys
import os

# 获取项目根目录（论文目录）的绝对路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# # 将根目录添加到sys.path
# sys.path.append(project_root)
import sys
sys.path.append('/hy-tmp')  # 手动加入 hy-tmp 的路径

from Data.HUST_dir import hust_L1, hust_L2, hust_L3, hust_L4
from data_generator.HUST_generator import HUST_preprocess
from torch.utils.data import DataLoader
from MTAGN_utils.train_utils import MyDataset


def load_hust_data(hust_data, Length=2048, Number=450, Overlap=True, Overlap_step=512, Shuffle=True,
                      Add_noise=False, SNR=-5, Normal=True, Rate=(150, 150, 150), is_source=True):
    """
    通用数据处理函数，适用于不同的 Chopper_dealing 数据集（L1, L2, L3, L4）
    """
    train_X = np.empty((0, Length), dtype=np.float32)
    train_Y1 = np.empty((1, 0), dtype=np.int64)
    train_Y2 = np.empty((1, 0), dtype=np.int64)
    valid_X = np.empty((0, Length), dtype=np.float32)
    valid_Y1 = np.empty((1, 0), dtype=np.int64)
    valid_Y2 = np.empty((1, 0), dtype=np.int64)
    test_X = np.empty((0, Length), dtype=np.float32)
    test_Y1 = np.empty((1, 0), dtype=np.int64)
    test_Y2 = np.empty((1, 0), dtype=np.int64)

    labels_map = [
        (0, 0),  # H_L* (正常状态或第一种类别、第一种严重度)
        (1, 1),  # B_H (第二种类别、第二种严重度)
        (1, 2),  # B_L (第二种类别、第三种严重度)
        (2, 1),  # C_L (第三种类别、第二种严重度)
        (2, 2),  # C_M (第三种类别、第三种严重度)
        (3, 1),  # I_H (第四种类别、第二种严重度)
        (3, 2),  # I_L (第四种类别、第三种严重度)
        (4, 1),  # 0_H (第五种类别、第二种严重度)
        (4, 2),  # 0_L (第五种类别、第三种严重度)
    ]

    if len(hust_data) != len(labels_map):
        raise ValueError(
            f"hust_data length ({len(hust_data)}) does not match labels_map length ({len(labels_map)})")

    # 2. 统一使用一个循环遍历所有文件
    for idx, file_path in enumerate(hust_data):
        label1_val, label2_val = labels_map[idx]  # 获取对应的标签

        trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 =HUST_preprocess(
            file_dir=file_path,
            data_len=Length,
            sample_num=Number,
            overlap=Overlap,
            overlap_step=Overlap_step,
            shuffle=Shuffle,
            add_noise=Add_noise,
            SNR=SNR,
            normalize=Normal,
            split_rate=Rate,
            label1=label1_val,
            label2=label2_val
        )

        train_X = np.vstack((train_X, trainX))
        train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
        valid_X = np.vstack((valid_X, validX))
        valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
        test_X = np.vstack((test_X, testX))
        test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)


    # 调整维度
    train_X = np.reshape(train_X, (-1, 1, Length))
    valid_X = np.reshape(valid_X, (-1, 1, Length))
    test_X = np.reshape(test_X, (-1, 1, Length))

    if is_source:
        return train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2
    else:
        return test_X, test_Y1, test_Y2


def HUST_L1(is_source=True):
    return load_hust_data(hust_L1, is_source=is_source)


def HUST_L2(is_source=True):
    return load_hust_data(hust_L2, is_source=is_source)


def HUST_L3(is_source=True):
    return load_hust_data(hust_L3, is_source=is_source)


def HUST_L4(is_source=True):
    return load_hust_data(hust_L4, is_source=is_source)


if __name__ == '__main__':
    # 你可以根据需求选择加载不同的层次数据
    train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2 = HUST_L1(is_source=True)
    test_X, test_Y1, test_Y2 = HUST_L1(is_source=False)

