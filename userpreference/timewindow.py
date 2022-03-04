"""
Qi Y,  Luo  J, Lin Gao,  Fu-Chun Zheng, Li Yu. User  Preference and Activity  Aware  Content  Sharing  in  Distribu
ted Wireless  D2D  Caching  Networks[C] 2020 IEEE/CIC International Conference on Communications in China (ICCC),
2020. (Accepted)

这篇论文提出了一个叫做EM （ Expectation Maximization）算法来学习用户偏好，并通过仿真证明了预测的准确性较高
采用滑动窗口法预测实时用户活跃程度，联合考虑用户偏好和实时用户活跃程度来预测实时文件流行度

用户偏好学习算法流程

t 时间
f 文件  F
k 用户  U
j 文件类型  Z

preference 用户偏好
active 用户活跃程度
popular 流行程度
like 喜好程度
probability 概率

为了生成用户的个人偏好概率，首先分别通过其对应的分布来生
成个人流行度分布和排名顺序，然后通过这两个分布来生成所需的概率

个人流行度分布:大小分布  个人类型流行度分布  基于类型的条件流行度分布

个人排名顺序的统计信息：大小分布  类型偏好概率  类型排名分布  文件排名分布
"""


# tm太亮了

# 用户u_k对文件f用户偏好可以表示为  （3-1）
def preference_k_f_t(active_k_t, like_j_k, popular_f_j):
    # TODO
    return


# 在第t个时间周期内用户u_k请求文件f的概率为 (3-2)
def probability_t_k_f(active_k_t, like_j_k, popular_f_j):
    return active_k_t * preference_k_f_t(active_k_t, like_j_k, popular_f_j)


# 用户整体偏好 （3-3）
def popular_f_t(active_t, preference_f_t):
    ans = 0
    for i in range(len(active_t)):
        ans += active_t[i] * preference_f_t[i]

    return ans




