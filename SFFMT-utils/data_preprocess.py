import torch
import numpy as np
from sklearn import preprocessing


def add_wgn(x, snr=5):
    """
    :param x: [num, data_len]
    :param snr: Signal Noise Ratio = 10log10(p_s/p_n)
    :return: signal + noise(d * [sqrt(p_n/p_d)])
    """
    num, len_x = x.shape
    d = np.random.randn(len_x)
    # power of random noise
    p_d = np.sum(d ** 2)
    # power of signal
    p_s = np.sum(np.power(x, 2), axis=1, keepdims=True)  # [num, 1]
    # power of noise
    p_n = p_s / (np.power(10, snr / 10))  # [num, 1]
    noise = np.sqrt(p_n / p_d) * d
    # print(10 * np.log10(p_s / (np.sum(np.power(noise, 2), axis=1, keepdims=True))))
    return (x + noise).astype('float32')


# def standardization(x):
#     """
#     input 二维numpy array, 对每个样本（每一行）标准化
#     第i行: x[i] = x[i] - mean(x[i]) / std(x[i])
#     :param x: [num, dim]
#     :return: [num, dim]
#     """
#     # x_out = preprocessing.scale(x, axis=1)  # default axis=0(列)
#     x_mean = np.mean(x, axis=1, keepdims=True)
#     x_std = np.std(x, axis=1, keepdims=True)
#     x_out = x - x_mean
#     x_out = x_out / x_std
#     return x_out


# def standardization(x):
#     """
#     输入：二维numpy数组，形状为[num_samples, feature_dim]
#     对每一行先标准化（减均值除以标准差），再归一化（min-max归一化到0-1）
#
#     :param x: [num_samples, feature_dim]
#     :return: 归一化后的数组，形状不变
#     """
#     # 标准化
#     x_mean = np.mean(x, axis=1, keepdims=True)
#     x_std = np.std(x, axis=1, keepdims=True)
#     x_out = (x - x_mean) / x_std
#
#     # 归一化（min-max归一化）
#     x_min = np.min(x_out, axis=1, keepdims=True)
#     x_max = np.max(x_out, axis=1, keepdims=True)
#     x_out = (x_out - x_min) / (x_max - x_min)
#
#     return x_out

def standardization(x):
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    x_out = (x - x_min) / (x_max - x_min + 1e-8)
    return x_out


