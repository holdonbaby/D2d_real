import numpy as np
import matplotlib.pyplot as plt
import random
import math


a=[ 50, 100 ,150 ,200, 250]
b=[0.62862239 ,0.38630159 ,0.29104522 ,0.23396765 ,0.19467652]
c=[0.5795285 , 0.337119  , 0.24997526 ,0.19885163, 0.16806038]
d=[0.4525061 , 0.27559677, 0.20984191 ,0.16860726, 0.14528958]
e=[0.38098735, 0.208313 ,  0.15113326, 0.12005075, 0.10005732]
f=[0.59644266 ,0.36563056, 0.27547895 ,0.22162462, 0.18518796]

for i in range(len(a)):
    b[i]*=a[i]
    c[i] *= a[i]
    d[i] *= a[i]
    e[i] *= a[i]
    f[i] *= a[i]



plt.title('x-user number , y-TH')
plt.plot(a,b)
plt.plot(a,c)
plt.plot(a,d)
plt.plot(a,e)
plt.plot(a,f)
plt.show()

print(b)
print(c)
print(d)
print(e)
print(f)
