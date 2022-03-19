import numpy as np
import matplotlib.pyplot as plt
#%%
#黄金分割法
rate = 0.618034
pointset = []
x = 0
y = 0
xxxx = 0.5
l_m = 0.7
def f(x):
    return 1.0 * (pow(x, 4) - 4 * pow(x, 3) - 6 * pow(x, 2) - 16 * x + 4)

def backtrace(f, a0, b0, accuracy):

    a = a0
    b = b0
    x2 = a + rate*(b - a)
    x1 = a + b - x2
    f2 = f(x2)
    f1 = f(x1)
    #print( x1, x2, '\n')
    arr = search(f, a, b, f1, f2, x1, x2, accuracy)
    return arr[1]
    #printFunc(f, a, b, arr[0], arr[1])

def search(f, a, b, f1, f2, x1, x2, accuracy):
    if f1 <= f2:
        if x2 - a < accuracy:
            x = x1
            y = f1
            #print( x, y)
            return (x, y)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + b - x1
            f2 = f(x2)
            #print(x1, x2, '\n')
            return search(f, a, b, f1, f2, x1, x2, accuracy)

    else:
        if b - x1 < accuracy:
            x = x2
            y = f2
            #print (x, y)
            return (x, y)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + b - x2
            f1 = f(x1)
            #print( x1, x2, '\n')
            return search(f, a, b, f1, f2, x1, x2, accuracy)


#绘制函数图像

def printFunc(f, a, b, x, y):
    t = np.arange(a, b, 0.01)
    s = f(t)
    plt.plot(t, s)
    plt.plot([x], [y], 'ro')
    plt.show()

def d2d_seg(f_m): #目标函数
    h_m=2*xxxx-2*f_m
    a=(1-l_m)*(f_m+(1-f_m)*(1-np.exp(-1*N*value*f_m)))
    b=l_m*((f_m+h_m)+(1-f_m-h_m)*(1-np.exp(-1*N*value*(f_m+h_m))))
    #print(a,b)
    #print(f_m,h_m)

    return  1*(a+b)
#%%
def d2d(a,x):
    return a-a*np.exp(-1.0*value* N * x)+a*x*np.exp(-1.0*value* N * x)

def zipf_distribute(alpha, file_number):
    ans = [0 for i in range(file_number)]
    sum = 0
    for i in range(1, file_number + 1):
        sum += 1 / pow(i, alpha)
    for i in range(1, file_number + 1):
        ans[i - 1] = (1 / pow(i, alpha)) / sum
    return ans

def all_d2d(results,f_func):
    ans=0
    for i in range(len(results)):

        ans+=f_func(zipf_n[i],results[i])
    return ans

def d2d_grad(a,x):
    return value*N*a*np.exp(-1.0*value* N * x)-a*x*value*np.exp(-1.0* N *value* x)+a*np.exp(-1.0*value* N * x)


def gd_d2d(results,f_grad):
    ans = np.copy(results)
    for i in range(len(results)):
        if ans[i]>=1:
            continue
        ans[i]=ans[i]+eta*f_grad(zipf_n[i],ans[i])
    return ans

def train_d2d(trainer,step=100,f_grad=None):
    results=np.zeros(M)
    for i in range(step):
        if results.sum()>cap:
            break
        results=np.copy(trainer(results,f_grad))
    return results
#%%
def d2d_gd():
    k = cap*M*10
    results = np.zeros(M)
    for i in range(k):
        max=0
        index=0
        for j in range(len(results)):
            if d2d(zipf_n[j],results[j]+lr)-d2d(zipf_n[j],results[j])>max:
                max =d2d(zipf_n[j],results[j]+lr)-d2d(zipf_n[j],results[j])
                index=j


        results[index]+=lr
    return results
#%%
#result = train_d2d(gd_d2d,100000,d2d_grad)

#%%

#%%
#所有的变量 N,M,alpha,lm
N_variable = np.arange(50,150,10)
M_variable = np.arange(50,150,10)
alpha_variable = np.arange(0.5,1.5,0.1)
lm_variable = np.arange(0.6,0.9,0.05)
C_variable = np.arange(3,6,1)
#%%
dim = 100
M=dim
c1 = 1.4
c2 = 1.4
max_gen = 500
size_pop =1000
V_max = 0.001
V_min = -0.001
pop_max = 1
pop_min = 0
w = 0.7
record = np.zeros(max_gen)
alpha = 1.0
zipf_n = zipf_distribute(alpha,dim)
lr = 1/(M*10)
value = (30**2)/(100**2)
N=100
eta=0.00001
cap =3
#%%
#重制默认值
M=100
alpha = 1.0
lr = 1/(M*10)
value = (30**2)/(100**2)
N=100
eta=0.00001
cap =3
#%%
#N结果向量存储







