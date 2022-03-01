
# 方法1  得到zipf分布的文件请求概率
def zipf_distribute(alpha, file_number):
    ans = [0 for i in range(file_number)]
    sum_p = 0
    for i in range(1, file_number + 1):
        sum_p += 1 / pow(i, alpha)
    for i in range(1, file_number + 1):
        ans[i - 1] = (1 / pow(i, alpha)) / sum_p
    return ans
