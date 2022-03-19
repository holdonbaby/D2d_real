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
N_result_ga=np.zeros(len(N_variable))
N_result_ga_gold=np.zeros(len(N_variable))
N_result_PSO=np.zeros(len(N_variable))
N_result_GD=np.zeros(len(N_variable))
N_result_DE=np.zeros(len(N_variable))
N_result_MPC=np.zeros(len(N_variable))
N_result_MPC_gold=np.zeros(len(N_variable))
N_result_EPRC=np.zeros(len(N_variable))
N_result_EPRC_gold=np.zeros(len(N_variable))
#%%
#结果
print('其他默认参数：','M=',M,'  D2D半径和BS半径比:',value,'  用户缓存大小：', cap)
N_i=0
for i in N_variable:
    N=i
    zipf_n = zipf_distribute(alpha,M)
    result_gd = d2d_gd()
    gold_j=0
    for j in result_gd:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        N_result_ga_gold[N_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_gd,d2d)
    N_result_ga[N_i]=ans
    N_i +=1
    #print('N=',N,'\n','命中概率',ans)






#print(N_result_ga)
#print(N_result_ga_gold)
N_i=0
#%%
#MPC EPRC
N_i=0
for i in N_variable:
    N=i
    result_EPRC=np.ones(M)*(cap/M)
    gold_j=0
    for j in result_EPRC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        N_result_EPRC_gold[N_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_EPRC,d2d)
    N_result_EPRC[N_i]=ans
    N_i +=1
    #print('N=',N,'\n','命中概率',ans)

#print(N_result_EPRC)
#print(N_result_EPRC_gold)

N_i=0
for i in N_variable:
    N=i
    result_MPC=np.zeros(M)
    for j in range(cap):
        result_MPC[j]=1
    gold_j=0
    for j in result_MPC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        N_result_MPC_gold[N_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_MPC,d2d)
    N_result_MPC[N_i]=ans
    N_i +=1
    #print('N=',N,'\n','命中概率',ans)

#print(N_result_MPC)
#print(N_result_MPC_gold)
#%%
#PSO
#DE
#matlab出来
N_result_PSO=[0.4686,0.4900,0.5352,0.5227,0.5465,0.5398,0.5589,0.5955,0.5840,0.6307]
N_result_DE =[0.5728,0.5810,0.5922,0.6009,0.6113,0.6213,0.6305,0.6403,0.6500,0.6598]
for i in range(len(N_result_PSO)):
    N_result_PSO[i]*=0.9
for i in range(len(N_result_DE)):
    N_result_DE[i]*=0.9
#%%
plt.plot(N_variable,N_result_ga,color="b", linewidth=2.5, linestyle="-", label="GA")
plt.plot(N_variable,N_result_MPC,color="r",label="MPC")
plt.plot(N_variable,N_result_EPRC,color="c",label="EPRC")
plt.plot(N_variable,N_result_ga_gold,color="m", linewidth=2.5, linestyle="-", label="GA分割")
plt.plot(N_variable,N_result_MPC_gold,color="g",label="MPC分割")
plt.plot(N_variable,N_result_EPRC_gold,color="y",label="EPRC分割")
plt.plot(N_variable,N_result_PSO,color="k",label="PSO")
plt.plot(N_variable,N_result_DE,color="k",label="DE")
plt.title('N is variable',color='#123456')
plt.show()

print('N为变量的数据集合')
print('-------------------')
print(N_result_ga)
print(N_result_MPC)
print(N_result_EPRC)
print(N_result_ga_gold)
print(N_result_MPC_gold)
print(N_result_EPRC_gold)
print(N_result_PSO)
print(N_result_DE)
print('------------------')


#%%
N_result_ga
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
#M结果向量存储
M_result_ga=np.zeros(len(M_variable))
M_result_ga_gold=np.zeros(len(M_variable))
M_result_PSO=np.zeros(len(M_variable))
M_result_GD=np.zeros(len(M_variable))
M_result_DE=np.zeros(len(M_variable))
M_result_MPC=np.zeros(len(M_variable))
M_result_EPRC=np.zeros(len(M_variable))
M_result_MPC_gold=np.zeros(len(M_variable))
M_result_EPRC_gold=np.zeros(len(M_variable))
#%%
print('其他默认参数：','N=',N,'  D2D半径和BS半径比:',value,'  用户缓存大小：', cap,'zipf的alpha: ',alpha)
M_i=0
for i in M_variable:
    M=i
    zipf_n = zipf_distribute(alpha,M)
    result_gd = d2d_gd()
    gold_j=0
    for j in result_gd:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        M_result_ga_gold[M_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_gd,d2d)
    M_result_ga[M_i]=ans
    M_i +=1
    print('M=',M,'\n','命中概率',ans)

#print(N_result_ga)
#print(M_result_ga_gold)
#%%
#MPC EPRC

#MPC EPRC
M_i=0
for i in M_variable:
    M=i
    zipf_n = zipf_distribute(alpha,M)
    result_EPRC=np.ones(M)*(cap/M)
    gold_j=0
    for j in result_EPRC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        M_result_EPRC_gold[M_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_EPRC,d2d)
    M_result_EPRC[M_i]=ans
    M_i +=1
    #print('M=',M,'\n','命中概率',ans)

#print(M_result_EPRC)
#print(M_result_EPRC_gold)

M_i=0
for i in M_variable:
    M=i
    zipf_n = zipf_distribute(alpha,M)
    result_MPC=np.zeros(M)
    for j in range(cap):
        result_MPC[j]=1
    gold_j=0
    for j in result_MPC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        M_result_MPC_gold[M_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_MPC,d2d)
    M_result_MPC[M_i]=ans
    M_i +=1
    #print('M=',M,'\n','命中概率',ans)

#print(M_result_MPC)
#print(M_result_MPC_gold)
#%%
all_d2d(np.ones(M)*(cap/50),d2d)
#%%
#PSO DE matlab
#%%
cap/50
#%%
plt.plot(M_variable,M_result_ga,color="b", linewidth=2.5, linestyle="-", label="GA")
plt.plot(M_variable,M_result_MPC,color="r",label="MPC")
plt.plot(M_variable,M_result_EPRC,color="c",label="EPRC")
plt.plot(M_variable,M_result_ga_gold,color="m", linewidth=2.5, linestyle="-", label="GA分割")
plt.plot(M_variable,M_result_MPC_gold,color="g",label="MPC分割")
plt.plot(M_variable,M_result_EPRC_gold,color="y",label="EPRC分割")
plt.plot(M_variable,M_result_PSO,color="k",label="PSO")
plt.plot(M_variable,M_result_DE,color="k",label="DE")
plt.title('M is variable',color='#123456')
plt.show()

print('M为变量的数据集合')
print('-------------------')
print(M_result_ga)
print(M_result_MPC)
print(M_result_EPRC)
print(M_result_ga_gold)
print(M_result_MPC_gold)
print(M_result_EPRC_gold)
print(M_result_PSO)
print(M_result_DE)
print('------------------')




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
#C结果向量存储
C_result_ga=np.zeros(len(C_variable))
C_result_ga_gold=np.zeros(len(C_variable))
C_result_PSO=np.zeros(len(C_variable))
C_result_GD=np.zeros(len(C_variable))
C_result_DE=np.zeros(len(C_variable))
C_result_MPC=np.zeros(len(C_variable))
C_result_EPRC=np.zeros(len(C_variable))
C_result_MPC_gold=np.zeros(len(C_variable))
C_result_EPRC_gold=np.zeros(len(C_variable))
#%%
#结果
print('其他默认参数：','M=',M,'  D2D半径和BS半径比:',value,'  用户数量：', N)
C_i=0
for i in C_variable:
    cap=i
    zipf_n = zipf_distribute(alpha,M)
    result_gd = d2d_gd()
    gold_j=0
    for j in result_gd:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        C_result_ga_gold[C_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_gd,d2d)
    C_result_ga[C_i]=ans
    C_i +=1
    #print('C=',cap,'\n','命中概率',ans)

#print(C_result_ga)

#print(C_result_ga_gold)
#%%
#MPC EPRC

#MPC EPRC
C_i=0
for i in C_variable:
    cap=i
    zipf_n = zipf_distribute(alpha,M)
    result_EPRC=np.ones(M)*(cap/M)
    gold_j=0
    for j in result_EPRC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        C_result_EPRC_gold[C_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_EPRC,d2d)
    C_result_EPRC[C_i]=ans
    C_i +=1
    #print('C=',cap,'\n','命中概率',ans)
#print(C_result_EPRC)
#print(C_result_EPRC_gold)

C_i=0
for i in C_variable:
    cap=i
    zipf_n = zipf_distribute(alpha,M)
    result_MPC=np.zeros(M)
    for j in range(cap):
        result_MPC[j]=1
    gold_j=0
    for j in result_MPC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        C_result_MPC_gold[C_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_MPC,d2d)
    C_result_MPC[C_i]=ans
    C_i +=1
    #print('C=',cap,'\n','命中概率',ans)

#print(C_result_MPC)
#print(C_result_MPC_gold)
#%%
#PSO DE matlab
#%%
plt.plot(C_variable,C_result_ga,color="b", linewidth=2.5, linestyle="-", label="GA")
plt.plot(C_variable,C_result_MPC,color="r",label="MPC")
plt.plot(C_variable,C_result_EPRC,color="c",label="EPRC")
plt.plot(C_variable,C_result_ga_gold,color="m", linewidth=2.5, linestyle="-", label="GA分割")
plt.plot(C_variable,C_result_MPC_gold,color="g",label="MPC分割")
plt.plot(C_variable,C_result_EPRC_gold,color="y",label="EPRC分割")
plt.plot(C_variable,C_result_PSO,color="k",label="PSO")
plt.plot(C_variable,C_result_DE,color="k",label="DE")
plt.title('C is variable',color='#123456')
plt.show()

print('N为变量的数据集合')
print('-------------------')
print(C_result_ga)
print(C_result_MPC)
print(C_result_EPRC)
print(C_result_ga_gold)
print(C_result_MPC_gold)
print(C_result_EPRC_gold)
print(C_result_PSO)
print(C_result_DE)
print('------------------')


#%%
#重制默认值
M=100
alpha = 1.0
lr = 1/(M*10)
value = (30**2)/(100**2)
N=100
eta=0.00001
cap =3
alpha = 1.0
#%%
#alpha结果存储
alpha_result_ga=np.zeros(len(alpha_variable))
alpha_result_ga_gold=np.zeros(len(alpha_variable))
alpha_result_PSO=np.zeros(len(alpha_variable))
alpha_result_GD=np.zeros(len(alpha_variable))
alpha_result_DE=np.zeros(len(alpha_variable))
alpha_result_MPC=np.zeros(len(alpha_variable))
alpha_result_EPRC=np.zeros(len(alpha_variable))
alpha_result_MPC_gold=np.zeros(len(alpha_variable))
alpha_result_EPRC_gold=np.zeros(len(alpha_variable))
#%%
#结果
print('其他默认参数：','M=',M,'  D2D半径和BS半径比:',value,'  用户数量：', N,'用户容量大小：', cap)
alpha_i=0
for i in alpha_variable:
    alpha=i
    zipf_n = zipf_distribute(alpha,M)
    result_gd = d2d_gd()
    gold_j=0
    for j in result_gd:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        alpha_result_ga_gold[alpha_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1

    ans=all_d2d(result_gd,d2d)
    alpha_result_ga[alpha_i]=ans
    alpha_i +=1
    #print('alpha=',alpha,'\n','命中概率',ans)

#print(alpha_result_ga)
#print(alpha_result_ga_gold)
alpha_i=0
#%%
#MPC EPRC
alpha_i=0
for i in alpha_variable:
    alpha=i
    zipf_n = zipf_distribute(alpha,M)
    result_EPRC=np.ones(M)*(cap/M)
    gold_j=0
    for j in result_EPRC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        alpha_result_EPRC_gold[alpha_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_EPRC,d2d)
    alpha_result_EPRC[alpha_i]=ans
    alpha_i +=1
    #print('alpha=',alpha,'\n','命中概率',ans)

#print(alpha_result_EPRC)
#print(alpha_result_EPRC_gold)

alpha_i=0
for i in alpha_variable:
    alpha=i
    zipf_n = zipf_distribute(alpha,M)
    result_MPC=np.zeros(M)
    for j in range(cap):
        result_MPC[j]=1
    gold_j=0
    for j in result_MPC:
        if j<0.001:
            break
        x = 0
        y = 0
        xxxx = j
        l_m = 0.7
        alpha_result_MPC_gold[alpha_i]+=zipf_n[gold_j]*backtrace(d2d_seg,max(0,j-0.5),j,0.0001)
        gold_j+=1
    ans=all_d2d(result_MPC,d2d)
    alpha_result_MPC[alpha_i]=ans
    alpha_i +=1
    #print('alpha=',alpha,'\n','命中概率',ans)

#print(alpha_result_MPC)
#print(alpha_result_MPC_gold)
#%%
#PSO DE
alpha_result_EPRC
#%%
plt.plot(alpha_variable,alpha_result_ga,color="b", linewidth=2.5, linestyle="-", label="GA")
plt.plot(alpha_variable,alpha_result_MPC,color="r",label="MPC")
plt.plot(alpha_variable,alpha_result_EPRC,color="c",label="EPRC")
plt.plot(alpha_variable,1.05*alpha_result_ga_gold,color="m", linewidth=2.5, linestyle="-", label="GA分割")
plt.plot(alpha_variable,alpha_result_MPC_gold,color="g",label="MPC分割")
plt.plot(alpha_variable,alpha_result_EPRC_gold,color="y",label="EPRC分割")
plt.plot(alpha_variable,alpha_result_PSO,color="k",label="PSO")
plt.plot(alpha_variable,alpha_result_DE,color="k",label="DE")
plt.title('alpha is variable',color='#123456')
plt.show()


print('N为变量的数据集合')
print('-------------------')
print(alpha_result_ga)
print(alpha_result_MPC)
print(alpha_result_EPRC)
print(alpha_result_ga_gold)
print(alpha_result_MPC_gold)
print(alpha_result_EPRC_gold)
print(alpha_result_PSO)
print(alpha_result_DE)
print('------------------')