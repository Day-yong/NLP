# --encoding:utf-8 --
"""实现viterbi算法"""

import common
import numpy as np


def viterbi(pi, A, B, Q, fetch_index_by_obs_seq=None):
    """计算观测序列"""
    # 0. 初始化
    # 初始化序列转换为索引的方法
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 初始化相关的参数值: n、m、T
    T = len(Q)
    n = len(A)
    n_range = range(n)
    # 存储delta值
    delta = np.zeros((T, n))
    # 存储状态值，pre_index[t][i]表示t时刻的状态为i，上一个最优状态(delta值最大的)为pre_index[t][i]
    pre_index = np.zeros((T, n), dtype=np.int)

    # 1. 计算t=1的时候delta的值
    for i in n_range:
        delta[0][i] = pi[i] * B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 2. 更新其它时刻的值
    for t in range(1, T):
        for i in n_range:
            # 当前时刻t的状态为i
            # a. 获取最大值
            max_delta = -1
            for j in n_range:
                # j表示的是上一个时刻的状态值
                tmp = delta[t - 1][j] * A[j][i]
                if tmp > max_delta:
                    max_delta = tmp
                    pre_index[t][i] = j

            # b. 更新值
            delta[t][i] = max_delta * B[i][fetch_index_by_obs_seq_f(Q, t)]

    # 3. 解码操作，查找到最大的结果值
    decode = [-1 for i in range(T)]
    # 先找最后一个时刻的delta最大值
    max_delta_index = 0
    for i in range(1, n):
        if delta[T - 1][i] > delta[T - 1][max_delta_index]:
            max_delta_index = i
    decode[T - 1] = max_delta_index
    # 再根据转移的路径(最大转移路径), 找出最终的链路
    for t in range(T - 2, -1, -1):
        max_delta_index = pre_index[t + 1][max_delta_index]
        decode[t] = max_delta_index
    return decode


if __name__ == '__main__':
    # 测试
    np.random.seed(28)
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

    # 开始计算
    state_seq = viterbi(pi, A, B, Q, common.convert_obs_seq_2_index)
    print("最终结果为:", end='')
    print(state_seq)
    state = ['盒子1', '盒子2', '盒子3']
    for i in state_seq:
        print(state[i], end='\t')
