import math
from random import random

import numpy as np



#DB为单位
E_D2D = 30
B_D2D = 20
E_BS = 46
B_BS = 20
N=100
d_0 =10
rou_c = 3*10**8/(2*10*9)
eta = 3.68

#DB为单位
SNR_min = 5

def db_to_normal(db):
    return 10**(db/10)

def path_model(distance):
    return 20*np.log10(4*np.pi*d_0/rou_c)+10*eta*np.log10(distance/d_0)


E_D2D_10 = db_to_normal(E_D2D)
B_D2D_10 = db_to_normal(B_D2D)/N
E_BS_10 = db_to_normal(E_BS)/1000
B_BS_10 = db_to_normal(B_BS)/1000

print(E_D2D_10)
print(B_D2D_10)
print(E_BS_10)
print(B_BS_10)

print(path_model(20))















