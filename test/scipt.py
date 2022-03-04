import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
mu = 30  # mean of distribution
sigma = 2  # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)


def mle(x):
    """
    极大似然估计
    :param x:
    :return:
    """
    u = np.mean(x)
    return u, np.sqrt(np.dot(x - u, (x - u).T) / x.shape[0])


print(mle(x))
num_bins = 100
plt.hist(x, num_bins)
plt.show()

