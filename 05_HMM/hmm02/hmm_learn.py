# --encoding:utf-8 --
"""HMM相关算法"""

import numpy as np

from common import log_sum_exp


def calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq=None):
    """
    计算前向概率α的值
    :param pi: 初始的状态随机概率矩阵, n*1 => 经过对数转换
    :param A:  状态转移矩阵, n*n => 经过对数转换
    :param B:  状态和观测值之间的转移矩阵, n*m => 经过对数转换
    :param Q:  观测值序列, 长度为T
    :param alpha:  前向概率矩阵
    :param fetch_index_by_obs_seq: 根据序列获取对应的索引值，可以为None；是一个接受两个参数的函数，第一个参数为序列，第二个参数为索引值
    :return:  返回alpha矩阵同时也更新传入的alpha矩阵参数
    """
    # 1. 初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        # 默认是使用ord函数将对应位置的字符转换为ASCII码，eg: ord('a')=97; ord('中')=20013
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 获取相关变量
    n = np.shape(A)[0]
    T = len(Q)

    # 3. 更新t=1时刻(初始时刻)对应的前向概率值
    for i in range(n):
        alpha[0][i] = pi[i] + B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 4. 更新t=2,3....T时刻对应的前向概率值
    tmp = [0 for i in range(n)]
    for t in range(1, T):
        # TODO: 开始更新时刻t、状态为i的前向概率值
        for i in range(n):
            # 4.1. 计算累加值
            for j in range(n):
                tmp[j] = alpha[t - 1][j] + A[j][i]

            # 4.2. 计算log_sum_exp的值
            alpha[t][i] = log_sum_exp(tmp)

            # 4.3. 累加状态和观测值之间的转移矩阵
            alpha[t][i] += B[i][fetch_index_by_obs_seq_f(Q, t)]

    # 5. 返回最终返回值
    return alpha


def calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq=None):
    """
    计算后向概率β的值
    :param pi: 初始的状态随机概率矩阵, n*1 => 经过对数转换
    :param A:  状态转移矩阵, n*n => 经过对数转换
    :param B:  状态和观测值之间的转移矩阵, n*m => 经过对数转换
    :param Q:  观测值序列, 长度为T
    :param beta: 后向概率矩阵
    :param fetch_index_by_obs_seq: 根据序列获取对应的索引值，可以为None；是一个接受两个参数的函数，第一个参数为序列，第二个参数为索引值
    :return:  返回beta矩阵同时也更新传入的beta矩阵参数
    """
    # 1. 初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        # 默认是使用ord函数将对应位置的字符转换为ASCII码，eg: ord('a')=97; ord('中')=20013
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 获取相关变量
    n = np.shape(A)[0]
    T = len(Q)

    # 3. 更新t=T时刻(初始时刻)对应的后向概率值
    for i in range(n):
        beta[T - 1][i] = 0

    # 4. 更新t=T-1,T-2....1时刻对应的前向概率值
    tmp = [0 for i in range(n)]
    for t in range(T - 2, -1, -1):
        # TODO: 更新时刻t对应状态为i的后向概率
        for i in range(n):
            # 4.1 计算累加值
            for j in range(n):
                tmp[j] = A[i][j] + beta[t + 1][j] + B[j][fetch_index_by_obs_seq_f(Q, t + 1)]

            # 4.2 计算log_sum_exp的值
            beta[t][i] = log_sum_exp(tmp)

    # 5. 结果返回
    return beta


def calc_gamma(alpha, beta, gamma):
    """
    根据alphe和beta的值计算gamma值
    最终结果保存在gamma矩阵中
    :param alpha: 传入的alpha矩阵 => 经过log转换后的
    :param beta:  传入的beta矩阵 => 经过log转换后的
    :param gamma: 传入的gamma矩阵，需要进行更新，最终结果是经过log转换后的
    :return: gamma矩阵
    """
    # 1. 获取相关变量
    T, n = np.shape(alpha)

    # 2. 遍历更新
    for t in range(T):
        # 2.1. 累加alpha和beta值(ppt上分子部分)
        for i in range(n):
            gamma[t][i] = alpha[t][i] + beta[t][i]

        # 2.2. 计算log_sum_exp的值（ppt上分母部分）
        lse = log_sum_exp(gamma[t])

        # 2.3. 计算最终结果
        for i in range(n):
            gamma[t][i] -= lse

    # 3. 返回最终结果
    return gamma


def calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq=None):
    """
    计算时刻t的时候状态为i，时刻t+1的时候状态为j的联合概率ksi
    :param alpha:  传入的alpha矩阵 => 经过log转换后的
    :param beta:  传入的beta矩阵 => 经过log转换后的
    :param A: 状态转移矩阵, n*n => 经过对数转换
    :param B:  状态和观测值之间的转移矩阵, n*m => 经过对数转换
    :param Q:  观测值序列, 长度为T
    :param ksi: 待求解的ksi概率矩阵，最终结果是需要经过log转换的
    :param fetch_index_by_obs_seq: 根据序列获取对应的索引值，可以为None；是一个接受两个参数的函数，第一个参数为序列，第二个参数为索引值
    :return: 返回ksi矩阵
    """
    # 1. 初始化
    # 初始化序列转换为索引的方法
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 变量获取
    T, n = np.shape(alpha)

    # 3. 开始迭代更新ksi矩阵
    tmp = np.zeros((n, n))
    for t in range(T - 1):
        # 3.1 计算t时刻状态为i，t+1时刻状态为j的概率值（ppt上的分子部分）
        for i in range(n):
            for j in range(n):
                tmp[i][j] = alpha[t][i] + A[i][j] + beta[t + 1][j] + B[j][fetch_index_by_obs_seq_f(Q, t + 1)]

        # 3.2 计算log_sum_exp的值(ppt上分母部分)
        lse = log_sum_exp(tmp.flat)

        # 3.3 计算最终的结果值
        for i in range(n):
            for j in range(n):
                ksi[t][i][j] = tmp[i][j] - lse

    # 4. 返回最终结果
    return ksi


def baum_welch(pi, A, B, Q, max_iter=10, fetch_index_by_obs_seq=None):
    """
    根据传入的初始概率矩阵(pi、A、B)以及观测序列Q，使用baum_welch算法进行迭代求解最终的pi、A、B的值；最大迭代次数默认为10；最终更新结果保存在传入的参数矩阵中(pi\A\B)
    :param pi: 初始的状态随机概率矩阵, n*1 => 经过对数转换
    :param A:  状态转移矩阵, n*n => 经过对数转换
    :param B:  状态和观测值之间的转移矩阵, n*m => 经过对数转换
    :param Q:  观测值序列, 长度为T
    :param max_iter: 最大迭代次数，默认为10
    :param fetch_index_by_obs_seq: 根据序列获取对应的索引值，可以为None；是一个接受两个参数的函数，第一个参数为序列，第二个参数为索引值
    :return:  返回(pi, A, B)
    """
    # 1. 初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        # 默认是使用ord函数将对应位置的字符转换为ASCII码，eg: ord('a')=97; ord('中')=20013
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 初始化相关参数
    T = len(Q)
    n = np.shape(A)[0]
    if len(np.shape(B)) == 1:
        m = 1
    else:
        m = np.shape(B)[1]
    alpha_ = np.zeros((T, n))
    beta_ = np.zeros((T, n))
    gamma_ = np.zeros((T, n))
    ksi_ = np.zeros((T - 1, n, n))

    # 3. 开始遍历求解
    for time in range(max_iter):
        # 3.1 分别计算在当前情况下的alphe、beta、gamma、ksi的值
        calc_alpha(pi, A, B, Q, alpha_, fetch_index_by_obs_seq_f)
        calc_beta(pi, A, B, Q, beta_, fetch_index_by_obs_seq_f)
        calc_gamma(alpha_, beta_, gamma_)
        calc_ksi(alpha_, beta_, A, B, Q, ksi_, fetch_index_by_obs_seq_f)

        # 3.2 更新pi的值
        for i in range(n):
            pi[i] = gamma_[0][i]

        # 3.3 更新A的值
        tmp1 = np.zeros(T - 1)  # 对应ppt上分子
        tmp2 = np.zeros(T - 1)  # 对应ppt上的分母
        for i in range(n):
            for j in range(n):
                # 3.3.1 获取所有时刻从状态i转移到状态j的概率值
                for t in range(T - 1):
                    tmp1[t] = ksi_[t][i][j]
                    tmp2[t] = gamma_[t][i]

                # 3.3.2 计算状态转移矩阵
                A[i][j] = log_sum_exp(tmp1) - log_sum_exp(tmp2)

        # 3.4 更新B的值
        tmp1 = np.zeros(T)  # 对应ppt上分子
        tmp2 = np.zeros(T)  # 对应ppt上的分母
        for i in range(n):
            for o in range(m):
                # 3.4.1 获取所有时刻位于状态i的概率
                valid = 0
                for t in range(T):
                    # a. 计算分子的值
                    if o == fetch_index_by_obs_seq_f(Q, t):
                        # 当前观测值和对应观测值一致，计算分子
                        tmp1[valid] = gamma_[t][i]
                        valid += 1
                    # b. 累积分母的值
                    tmp2[t] = gamma_[t][i]

                # 3.4.2 更新状态转移矩阵的值B
                if valid == 0:
                    # 表示没有从i和o的观测值转移矩阵
                    B[i][o] = 0
                else:
                    B[i][o] = log_sum_exp(tmp1[:valid]) - log_sum_exp(tmp2)

    # 4. 返回最终结果
    return (pi, A, B)


def viterbi(pi, A, B, Q, fetch_index_by_obs_seq=None):
    """
    计算观测序列
    :param pi: 初始的状态随机概率矩阵, n*1 => 经过对数转换
    :param A:  状态转移矩阵, n*n => 经过对数转换
    :param B:  状态和观测值之间的转移矩阵, n*m => 经过对数转换
    :param Q:  观测值序列, 长度为T
    :param fetch_index_by_obs_seq: 根据序列获取对应的索引值，可以为None；是一个接受两个参数的函数，第一个参数为序列，第二个参数为索引值
    :return: 返回decode序列
    """
    # 1.初始化
    # 初始化序列转换为索引的方法
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 相关参数初始化
    T = len(Q)
    n = np.shape(A)[0]
    delta = np.zeros((T, n))
    # 存储的是上一个最优状态值,eg: pre_optima_index[t][i]表示时刻t对应状态为i，此时上一个时刻的最优状态（delta最大）为pre_optima_index[t][i]
    pre_optima_index = np.zeros((T, n), dtype=np.int)

    # 3 计算t=1时刻的delta值
    for i in range(n):
        delta[0][i] = pi[i] + B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 4 计算t=2、3、4.....时刻的delta值
    for t in range(1, T):
        for i in range(n):
            # 4.1 获取最大的值以及对应的最优索引位置
            max_delta = delta[t - 1][0] + A[0][i]
            optima_index = 0
            for j in range(1, n):
                tmp_delta = delta[t - 1][j] + A[j][i]
                if max_delta < tmp_delta:
                    max_delta = tmp_delta
                    optima_index = j

            # 4.2 计算最终的delta值以及最优索引位置
            delta[t][i] = max_delta + B[i][fetch_index_by_obs_seq_f(Q, t)]
            pre_optima_index[t][i] = optima_index

    # 5. 解码操作，查找到最大的结果值（回溯找最大的路径）
    decode = [-1 for i in range(T)]
    # 先找最后一个时刻的delta最大值
    max_delta_index = 0
    for i in range(1, n):
        if delta[T - 1][i] > delta[T - 1][max_delta_index]:
            max_delta_index = i
    decode[T - 1] = max_delta_index
    # 再根据转移的路径(最大转移路径), 找出最终的链路
    for t in range(T - 2, -1, -1):
        max_delta_index = pre_optima_index[t + 1][max_delta_index]
        decode[t] = max_delta_index

    # 6. 返回最终结果
    return decode
