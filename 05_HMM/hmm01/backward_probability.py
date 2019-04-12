# --encoding:utf-8 --
"""计算后向概率"""

import common
import numpy as np


def calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq=None):
    """
    计算后向概率β的值
    pi：初始的随机概率值
    A：状态转移矩阵
    B: 状态和观测值之间的转移矩阵
    Q: 观测值列表
    beta：后向概率beta矩阵
    fetch_index_by_obs_seq: 根据序列获取对应索引值，可以为空
    NOTE:
        1. ord函数的含义是将一个单个的字符转换为数字, eg: ord('a') = 97; ord('中')=20013；底层其实是将字符转换为ASCII码；
        2. 最终会直接更新参数中的beta对象
    """
    # 0. 初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 1. 初始一个状态类别的顺序
    n = len(A)
    n_range = range(n)
    T = len(Q)

    # 2. 更新初值(t=T)
    for i in n_range:
        beta[T - 1][i] = 1

    # 3. 迭代更新其它时刻
    tmp = [0 for i in n_range]
    for t in range(T - 2, -1, -1):
        for i in n_range:
            # 1. 计算到下一个时刻t+1的概率值
            for j in n_range:
                tmp[j] = A[i][j] * beta[t + 1][j] * B[j][fetch_index_by_obs_seq_f(Q, t + 1)]

            # 2. 更新beta的值
            beta[t][i] = np.sum(tmp)


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
    # 开始计算
    calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    # 输出最终结果
    print(beta)

    # 计算最终概率值：
    p = 0
    for i in range(len(A)):
        p += pi[i] * B[i][common.convert_obs_seq_2_index(Q, 0)] * beta[0][i]
    print(Q, end="->出现的概率为:")
    print(p)
