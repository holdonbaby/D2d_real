from typing import Sized
import numpy as np
import math
import random
from matplotlib import pyplot as plt

"""
parameter:{
    terminal_number  用户终端个数
    radius_ 基站半径的平方
    communication_maxR  用户终端最大可d2d通信距离
    cache_Size  用户缓存空间大小
    file_Numbers    文件个数
    user_request_dummy_rate     虚拟用户请求文件概率
    zipf_alpha      zipf分布的参数   
}

new_parameter_01背包:{
    file_size  文件大小
    file_can_be_size    文件可以是的大小
}

methods:{
    random_point  
    candi_terminal
    hit_rate
    cache_which_file
    zipf_distribute
}
"""


# 方法1  得到zipf分布的文件请求概率
def zipf_distribute(alpha, file_number):
    ans = [0 for i in range(file_number)]
    sum = 0
    for i in range(1, file_number + 1):
        sum += 1 / pow(i, alpha)
    for i in range(1, file_number + 1):
        ans[i - 1] = (1 / pow(i, alpha)) / sum
    return ans


# 方法2  在圆环形内生成随机terminal的坐标
def random_point(car_num, radius):
    x_index = []
    y_index = []
    for i in range(1, car_num + 1):
        theta = random.random() * 2 * np.pi
        r = random.uniform(0, radius)
        x = math.cos(theta) * (r ** 0.5)
        y = math.sin(theta) * (r ** 0.5)
        x_index.append(x)
        y_index.append(y)

    return x_index, y_index


# 方法3  得到可以相互通信的用户集合
def candi_terminal(x_label, y_label, r_max):
    x = len(x_label)
    y = len(y_label)
    terminal_can_com = [[0 for i in range(x)] for j in range(y)]
    for i in range(0, x):
        for j in range(0, y):
            d = pow((x_label[i] - x_label[j]), 2) + pow((y_label[i] - y_label[j]), 2)
            if d <= pow(r_max, 2):
                terminal_can_com[i][j] = 1
    return terminal_can_com


# 方法4   得到基站内所有用户平均的请求命中概率
def hit_rate(user_number, file_number, is_cache_file, candidate_user, user_request_file_rate):
    ans = 0
    for user in range(user_number):
        for file in range(file_number):
            for candi in range(user_number):
                if is_cache_file[candi][file] == 1 & candidate_user[user][candi] == 1:
                    ans += user_request_file_rate
                    break
    return ans / user_number

    # for i in range(user_number):
    #     for file in range(len(is_cache_file[i])):
    #         if  is_cache_file[i][file]==1:
    #             for user__ in range(len(candidate_user[i])):
    #                 if candidate_user[i][user__]==1 & is_cache_file[i][file]==1:
    #                     ans+= user_request_file_rate*is_cache_file[i][file]*candidate_user[i][user__]
    #                     break


# 方法5
def cache_which_file(user_number, cache_size, file_number):
    is_cache_file = [[0 for i in range(user_number)] for j in range(file_number)]

    for j in range(user_number):
        for i in range(file_number):
            if sum(is_cache_file[j]) >= cache_size + j:
                break
            if user_number != 1:
                is_cache_file[j][i] = 1

    print(is_cache_file)
    return is_cache_file


# 方法6    0-1背包问题得到最大效能
def dp_01(file_size, cache_size, file_utility) -> list:
    dp = [[0 for i in range(len(file_size) + 1)] for j in range(cache_size + 1)]
    for i in range(1, len(file_size) + 1):
        for j in range(1, cache_size + 1):
            if j - file_size[i] <= 0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], file_utility[i - 1] + dp[i - 1][j - file_size[i - 1]])
    return dp


# 方法7     从上述最大效能 倒推 存的哪些文件。 与方法6连用。
def cache_which_file_01(dp, file_size, cache_size):
    cache_index = []
    k = len(file_size)
    l = cache_size
    for i in range(100):
        if dp[k][l] == dp[k - 1][l]:
            k = k - 1
        else:
            cache_index.append(k - 1)
            l -= file_size[k - 1]
            k = k - 1

        if k == 0:
            break

    return cache_index


def n_is_or_not_hit_m(cache_set, n, m, n_communication_set_n_):
    if cache_set[n][m] == 1:
        return 1

    for i in range(len(n_communication_set_n_)):
        if cache_set[n_communication_set_n_[i]][m] == 1:
            return 1

    return 0


# 公式（1）的实现 用户n的自命中概率
def n_hit_possible_self(cache_set, n, m_request_set):
    P = 0
    for i in range(len(m_request_set[n])):
        P += cache_set[n][i] * m_request_set[n][i]

    return P


# 公式(2)的实现  用户n的d2d命中概率
def n_hit_possible_d2d(cache_set, n, m_request_set, n_communication_set_n_):
    P = 0
    for i in range(len(m_request_set[n])):
        if cache_set[n][i] == 1:
            continue
        for j in range(len(n_communication_set_n_)):
            if n_communication_set_n_ == n:
                continue
            if cache_set[n_communication_set_n_[j]][i] == 1:
                P += m_request_set[n][i]
                break

    return P


# 公式(3)的实现
def n_hit_possbile_no_bs(cache_set, n, m_request_set, n_communication_set_n_):
    return n_hit_possible_self(cache_set, n, m_request_set) + n_hit_possible_d2d(cache_set, n, m_request_set,
                                                                                 n_communication_set_n_)


# 系统的平均命中概率，这个是我们要得到的最大值
def all_hit_possible(cache_set, n_communication_set_n_, m_request_set):
    P = 0
    for i in range(len(cache_set)):
        P += n_hit_possbile_no_bs(cache_set, i, m_request_set, n_communication_set_n_[i])

    return P / (len(cache_set))


# 生成 n_cache清空后 所有文件的被他缓存的价值矩阵
def clear_n_cache_set_all_file_value(cache_set, n, n_communication_set_n_, m_request_set):
    cache_set[n] = [0 for i in range(len(m_request_set[n]))]
    file_value = [0 for i in range(len(m_request_set[n]))]

    for h in range(len(file_value)):
        file_value[h] += m_request_set[n][h]
        for j in range(len(n_communication_set_n_)):
            n_ = n_communication_set_n_[j]
            if n_is_or_not_hit_m(cache_set, n_, h, n_communication_set_n_) == 0:
                file_value[h] += m_request_set[n_][h]

    return file_value


def reset_n_local_cache(file_value, file_size, cache_size, cache_set):
    dp = [[0 for i in range(cache_size + 1)] for j in range(len(file_size) + 1)]
    for i in range(1, len(file_size) + 1):
        for j in range(1, cache_size + 1):
            if j - file_size[i - 1] <= 0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], file_value[i - 1] + dp[i - 1][j - file_size[i - 1]])

    xxx = cache_which_file_01(dp, file_size, cache_size)
    for i in range(len(xxx)):
        cache_set[xxx[i]] = 1


# terminal numbers
# radius
# 最大通信距离
# 最大缓存空间
# 基站属性
terminal_number = 60
file_Numbers = 30
R_max = 100
radius_ = pow(R_max, 2)

# 用户属性
communication_maxR = 20
cache_Size = 15
user_request_dummy_rate = 0.01

# 文件属性
zipf_alpha = 1.5
file_size = [0 for i in range(file_Numbers)]
for i in range(len(file_size)):
    file_size[i] = random.randint(1, 5)
file_required_by_terminal = [[0 for i in range(file_Numbers)] for j in range(terminal_number)]
for j in range(len(file_required_by_terminal)):
    file_required_by_terminal[j] = zipf_distribute(random.uniform(1, 2), file_Numbers)
    random.shuffle(file_required_by_terminal[j])
print(file_required_by_terminal[1])
print(file_required_by_terminal[2])

# 用户地理位置集合
x_set, y_set = random_point(terminal_number, radius_)

# 用户通信候选用户集合 user_can_communication_set[n]表示 用户n的候选集合
user_can_communication_set = candi_terminal(x_set, y_set, communication_maxR)

# 用户缓存文件集合
is_cache_file_set = [[0 for i in range(file_Numbers)] for j in range(terminal_number)]

for k in range(10):
    for n in range(terminal_number):
        value = clear_n_cache_set_all_file_value(is_cache_file_set, n, user_can_communication_set[n],
                                                 file_required_by_terminal)
        reset_n_local_cache(value, file_size, cache_Size, is_cache_file_set[n])
    print(is_cache_file_set[1])
    print(is_cache_file_set[0])
    print(all_hit_possible(is_cache_file_set, user_can_communication_set, file_required_by_terminal))

# plt.title("user terminal distribute version I") 
# plt.scatter(x_set, y_set, marker='o', label="terminal")
# plt.scatter(0, 0, marker='^', label="base station")
# plt.legend(loc='best')
# #plt.plot(0,0,"^",color="red")
# plt.show()


'''
file_tag=[i for i in range(file_Numbers)]
str= "zipf _distribute  file_number:%d  zipf_alpha:%10.3f" %(file_Numbers,zipf_alpha)
plt.title(str)
plt.scatter(file_tag,file_require,marker='^',label='terminal')
plt.show()
'''
