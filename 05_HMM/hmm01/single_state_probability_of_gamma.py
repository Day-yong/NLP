# --encoding:utf-8 --
"""计算gama值,即给定模型lambda和观测序列Q的时候，时刻t对应状态i的概率值"""

import common
import forward_probability as forward
import backward_probability as backward
import numpy as np


def calc_gamma(alpha, beta, gamma):
    """
    根据alphe和beta的值计算gamma值
    最终结果保存在gamma矩阵中
    """
    T = len(alpha)
    n_range = range(len(alpha[0]))
    tmp = [0 for i in n_range]
    for t in range(T):
        # 累加t时刻对应的所有状态值的前向概率和后向概率，从而计算分母
        for i in n_range:
            tmp[i] = alpha[t][i] * beta[t][i]
        sum_alpha_beta_of_t = np.sum(tmp)

        # 更新gamma值
        for i in n_range:
            gamma[t][i] = tmp[i] / sum_alpha_beta_of_t


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
    beta = np.zeros((len(Q), len(A)))
    alpha = np.zeros((len(Q), len(A)))
    gamma = np.zeros((len(Q), len(A)))

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
    calc_gamma(alpha, beta, gamma)
    # 输出最终结果
    print("gamma矩阵:")
    print(gamma)

    # 选择每个时刻最大的概率作为预测概率
    print("各个时刻最大概率的盒子为:", end='')
    index = ['盒子1', '盒子2', '盒子3']
    for p in gamma:
        print(index[p.tolist().index(np.max(p))], end="\t")
