"""
用户位置信息
这个有两种分布方式
第一种 泊松分布
第二种 随机分布
基站一般都是圆形的

分布又有两种分布的类型
具体分布 每个用户都有具体的位置信息
概率分布 不会给用户具体的信息，而是以概率的方式的来推断用户周围有多少个可以进行通信的用户
"""


# 随机分布
# 变量有用户数量n 基站的通信半径r 返回的则是用户位置的横坐标和纵坐标
from keyan.mypaper import random_point


def random_local(n, r):
    return random_point(n,r)


# 泊松分布
# 变量有用户数量n 基站的通信半径r 返回的则是用户位置的横坐标和纵坐标
def poisson_local(n, r):
    return
