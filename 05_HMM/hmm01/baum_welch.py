# --encoding:utf-8 --
"""进行HMM参数学习的算法"""

import common
import forward_probability as forward
import backward_probability as backward
import single_state_probability_of_gamma as single
import continuous_state_probability_of_ksi as continuous
import numpy as np


def baum_welch(pi, A, B, Q, max_iter=3, fetch_index_by_obs_seq=None):
    """
    根据传入的初始概率矩阵(pi、A、B)以及观测序列Q，使用baum_welch算法进行迭代求解最终的pi、A、B的值；
    最大迭代次数默认为3；最终更新结果保存在传入的参数矩阵中(pi\A\B)
    """
    # 0. 初始化相关变量
    # 初始化序列转换为索引的方法
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 初始化相关的参数值: n、m、T
    T = len(Q)
    n = len(A)
    m = len(B[0])
    alpha = np.zeros((T, n))
    beta = np.zeros((T, n))
    gamma = np.zeros((T, n))
    ksi = np.zeros((T - 1, n, n))
    n_range = range(n)
    m_range = range(m)
    t_range = range(T)
    t_1_range = range(T - 1)

    # 1. 迭代更新(EM算法思想类型)
    for time in range(max_iter):
        # a. 在当前的pi，A，B的情况下对观测序列Q分别计算alpha、beta、gamma和ksi
        forward.calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq_f)
        backward.calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq_f)
        single.calc_gamma(alpha, beta, gamma)
        continuous.calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq_f)

        # b. 更新pi、A、B的值
        # b.1. 更新pi值
        for i in n_range:
            pi[i] = gamma[0][i]

        # b.2. 更新状态转移矩阵A的值
        tmp1 = np.zeros(T - 1)
        tmp2 = np.zeros(T - 1)
        for i in n_range:
            for j in n_range:
                # 获取所有时刻从状态i转移到状态j的值
                for t in t_1_range:
                    tmp1[t] = ksi[t][i][j]
                    tmp2[t] = gamma[t][i]

                # 更新状态i到状态j的转移概率
                A[i][j] = np.sum(tmp1) / np.sum(tmp2)

        # b.3. 更新状态和观测值之间的转移矩阵
        for i in n_range:
            for k in m_range:
                tmp1 = np.zeros(T)
                tmp2 = np.zeros(T)
                # 获取所有时刻从状态i转移到观测值k的概率和
                number = 0
                for t in t_range:
                    if k == fetch_index_by_obs_seq_f(Q, t):
                        # 如果序列Q中时刻t对应的观测值就是k，那么进行统计这个时刻t为状态i的概率值
                        tmp1[t] = gamma[t][i]
                        number += 1

                    tmp2[t] = gamma[t][i]

                # 更新状态i到观测值k之间的转移概率
                if number == 0:
                    # 没有转移，所以为0
                    B[i][k] = 0
                else:
                    # 有具体值，那么进行更新操作
                    B[i][k] = np.sum(tmp1) / np.sum(tmp2)


if __name__ == '__main__':
    # 测试
    np.random.seed(28)
    pi = np.random.randint(1, 10, 3)
    pi = pi / np.sum(pi)
    A = np.random.randint(1, 10, (3, 3))
    A = A / np.sum(A, axis=1).reshape((-1, 1))
    B = np.random.randint(1, 5, (3, 2))
    B = B / np.sum(B, axis=1).reshape((-1, 1))

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

    # 观测序列
    Q = '白黑白白黑'

    print("随机初始的概率矩阵:")
    print("pi初始状态概率矩阵:")
    print(pi)
    print("\n状态转移矩阵A:")
    print(A)
    print("\n状态和观测值之间的转移矩阵B:")
    print(B)

    # 开始计算
    baum_welch(pi, A, B, Q, max_iter=10, fetch_index_by_obs_seq=common.convert_obs_seq_2_index)

    # 输出最终结果
    print("\n\n\n最终计算出来的结果:")
    print("pi初始状态选择矩阵:")
    print(pi)
    print("\n状态转移矩阵A:")
    print(A)
    print("\n状态和观测值之间的转移矩阵B:")
    print(B)
