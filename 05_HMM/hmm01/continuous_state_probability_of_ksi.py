# --encoding:utf-8 --
"""计算两个连续状态的联合概率值可西/可赛值"""

import common
import forward_probability as forward
import backward_probability as backward
import single_state_probability_of_gamma as single
import numpy as np


def calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq=None):
    """
    计算时刻t的时候状态为i，时刻t+1的时候状态为j的联合概率ksi
    alpha：对应的前向概率值
    beta：对应的后向概率值
    A：状态转移矩阵
    B: 状态和观测值之间的转移矩阵
    Q: 观测值列表
    ksi：待求解的ksi矩阵
    fetch_index_by_obs_seq: 根据序列获取对应索引值的函数，可以为空
    NOTE:
        1. ord函数的含义是将一个单个的字符转换为数字, eg: ord('a') = 97; ord('中')=20013；底层其实是将字符转换为ASCII码；
        2. 最终会直接更新参数中的ksi矩阵
    """
    # 0. 初始化
    # 初始化序列转换为索引的方法
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 初始化相关的参数值: n、T
    T = len(alpha)
    n = len(A)

    # 1. 开始迭代更新
    n_range = range(n)
    tmp = np.zeros((n, n))

    for t in range(T - 1):
        # 1. 计算t时刻状态为i，t+1时刻状态为j的概率值
        for i in n_range:
            for j in n_range:
                tmp[i][j] = alpha[t][i] * A[i][j] * B[j][fetch_index_by_obs_seq_f(Q, t + 1)] * beta[t + 1][j]

        # 2. 计算t时候的联合概率和
        sum_pro_of_t = np.sum(tmp)

        # 2. 计算时刻t时候的联合概率ksi
        for i in n_range:
            for j in n_range:
                ksi[t][i][j] = tmp[i][j] / sum_pro_of_t


if __name__ == '__main__':
    # 测试
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    Q = '白黑白白黑'
    T = len(Q)
    n = len(A)
    beta = np.zeros((T, n))
    alpha = np.zeros((T, n))
    gamma = np.zeros((T, n))
    ksi = np.zeros((T - 1, n, n))

    # 开始计算
    # 1. 计算beta
    backward.calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    # 输出最终结果
    print("beta矩阵:")
    print(beta)
    # 2. 计算alpha
    forward.calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    # 输出最终结果
    print("alpha矩阵:")
    print(alpha)
    # 3. 计算gamm矩阵
    single.calc_gamma(alpha, beta, gamma)
    # 输出最终结果
    print("gamma矩阵:")
    print(gamma)
    # 4. 计算ksi矩阵
    calc_ksi(alpha, beta, A, B, Q, ksi, common.convert_obs_seq_2_index)
    # 输出最终结果
    print("ksi矩阵:")
    print(ksi)
