import numpy as np
import matplotlib.pyplot as plt
import random
import math


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


x_set,y_set = random_point(100,100**2)

theta = np.linspace(0, 2 * np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
plt.plot(x,y)
plt.title("user terminal distribute")
plt.scatter(x_set, y_set, marker='o', label="terminal")
plt.scatter(0, 0, marker='^', label="Base Station")
plt.legend(loc='best')
#plt.plot(0,0,"^",color="red")
plt.show()