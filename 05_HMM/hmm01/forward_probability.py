# --encoding:utf-8 --
"""前向概率计算"""

import common
import numpy as np


def calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq=None):
    """
    计算前向概率α的值
    pi：初始的随机概率值
    A：状态转移矩阵
    B: 状态和观测值之间的转移矩阵
    Q: 观测值列表
    alpha：前向概率alpha矩阵
    fetch_index_by_obs_seq: 根据序列获取对应索引值，可以为空
    NOTE:
        1. ord函数的含义是将一个单个的字符转换为数字, eg: ord('a') = 97; ord('中')=20013；底层其实是将字符转换为ASCII码；
        2. 最终会直接更新参数中的alpha对象
    """
    # 0. 初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 1. 初始一个状态类别的顺序
    n = len(A)
    n_range = range(n)

    # 2. 更新初值(t=1)
    for i in n_range:
        alpha[0][i] = pi[i] * B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 3. 迭代更新其它时刻
    T = len(Q)
    tmp = [0 for i in n_range]
    for t in range(1, T):
        for i in n_range:
            # 1. 计算上一个时刻t-1累积过来的概率值
            for j in n_range:
                tmp[j] = alpha[t - 1][j] * A[j][i]

            # 2. 更新alpha的值
            alpha[t][i] = np.sum(tmp) * B[i][fetch_index_by_obs_seq_f(Q, t)]


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
    alpha = np.zeros((len(Q), len(A)))
    # 开始计算
    calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    # 输出最终结果
    print(alpha)

    # 计算最终概率值：
    p = 0
    for i in alpha[-1]:
        p += i
    print(Q, end="->出现的概率为:")
    print(p)
