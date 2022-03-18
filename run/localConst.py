import math
from random import random

import numpy as np


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








