import numpy as np
import matplotlib.pyplot as plt
import random
import math





R_D2D_variable = np.arange(10,60,10)
y_1 = np.zeros(len(R_D2D_variable))
y_2 = np.zeros(len(R_D2D_variable))
y_3 = np.zeros(len(R_D2D_variable))
y_4 = np.zeros(len(R_D2D_variable))
y_5 = np.zeros(len(R_D2D_variable))

for variable in range(len(R_D2D_variable)):


    #%%
    #设置默认值
    N=100
    M=200
    R_D2D=R_D2D_variable[variable]
    R_BS=100
    S=12
    #%%
    # 功率和带宽的单位都为DB
    E_D2D = 20
    E_BS = 20/100
    B_BS = 20/100
    B_D2D = 20

    #
    v_min = np.log2(1 + 3.16)

    # 吞吐量
    TH_D2D = v_min * B_D2D
    TH_BS = TH_D2D * 1 / 4
    TH_self = TH_D2D * 3
    #%% md

    #%%
    #随机化的local message function
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
    #%% md

    #%%
    x,y = random_point(N,R_BS**2)
    # plt.title("user terminal distribute version I")
    # plt.scatter(x, y, marker='o', label="terminal")
    # plt.scatter(0, 0, marker='^', label="base station")
    # plt.legend(loc='best')
    # #plt.plot(0,0,"^",color="red")
    # plt.show()
    #%% md

    #%% md

    #%%
    #得到用户的 第一个表 可通信表 1
    # 用户i的表为communication[i,:]
    communication = np.zeros((N,N))
    for i in range(len(communication)):
        for j in range(len(communication)):
            if (x[i]-x[j])**2+(y[i]-y[j])**2<=R_D2D**2:
                communication[i][j]=1
    #%%
    communication[0].sum()-1
    #%%
    #初始化用户的存储表 2
    cache = np.zeros((N,M))
    #这个表会在后续循环中更新
    #%%
    #初始化用户的命中表 3
    #这个表是由用户自身的存储表和用户可通行表中其他用户的存储表相加得到的
    hit_nums = np.zeros((N,M))
    #%%
    #初始化用户的请求表 4
    #是由图神经网络预测得到的，先用随机值代替
    rate = 5*np.random.random((N,M))
    request = np.zeros((N,M))
    for i in range(len(rate)):
        for j in range(len(rate[0])):
            request[i][j]=rate[i][j]/rate[i].sum()

    #%%
    #初始化权重表5
    weight = np.zeros((N,3))
    weight[:,0]=TH_self
    weight[:,1]=TH_D2D
    weight[:,2]=TH_BS

    weight_P = np.zeros((N,3))
    weight_P[:, 0] = 1
    weight_P[:, 1] = 1
    weight_P[:, 2] = 0

    SELF = 0
    D2D = 1
    BASE = 2

    # 1 是 self 的权重，2 是 d2d 的权重  3是 BS的权重
    #如果是 命中概率最高， 那么 self =1  d2d =1  BS =0 即可
    weight
    #%% md

    #%%
    #循环前一些的值说明

    #表示总的效能
    U=0
    #K表示循环论数 一定是 N的 K次方  一般里说  k=5就行了，不会比这个更高的
    K=5
    #把每一轮循环的结果 存起来
    ans_U = np.zeros(K)
    U_self=0
    U_D2D = 0
    U_BASE = 0

    for i in range(K):
        random_user_set = np.arange(0,N,1)
        np.random.shuffle(random_user_set)
        print('这是第的',i*N,'轮迭代')
        #挑选一个用户 index 来是的他的缓存空间存储的对象可以使得 U全局 最大化
        for index in random_user_set:
            cache_U = np.zeros((M,2)) #用户 index 存入文件m 需要一个 表记录
            cache_U_0 = np.zeros((M,2))
            #cache_tmp矩阵表示临时表， index行全1
            cache_tmp =cache.copy()
            cache_tmp[index,:]=np.ones(M)
            cache_tmp_0 = cache.copy()
            cache_tmp_0[index,:]=np.zeros(M)
            #
            hit_nums = np.zeros((N,M))
            hit_nums_0 = np.zeros((N,M))
            for n in range(N):
                if communication[index,n]==0: #过滤所有的不能于index 通信的n
                    continue
                for n_nearby in range(N):
                    if communication[n,n_nearby]==0:
                        continue
                    hit_nums[n,:]+=cache_tmp[n_nearby,:] #得到n所有的hit数量
            for m in range(M):
                cache_U[m,1]=m
                for n in range(N):
                    if communication[index,n]==0: #过滤所有的不能于index 通信的n
                        continue
                    if hit_nums[n,m]>=1:
                        if cache_tmp[n,m]==1:
                            cache_U[m,0]+=request[n,m]*weight[n,SELF]
                        else:
                            cache_U[m,0]+=request[n,m]*weight[n,D2D]
                    else:
                        cache_U[m,0]+=request[n,m]*weight[n,BASE]

            for n in range(N):
                if communication[index,n]==0: #过滤所有的不能于index 通信的n
                    continue
                for n_nearby in range(N):
                    if communication[n,n_nearby]==0:
                        continue
                    hit_nums_0[n,:]+=cache_tmp_0[n_nearby,:] #得到n所有的hit数量
            for m in range(M):
                cache_U_0[m,1]=m
                for n in range(N):
                    if communication[index,n]==0: #过滤所有的不能于index 通信的n
                        continue
                    if hit_nums_0[n,m]>=1:
                        if cache_tmp_0[n,m]==1:
                            cache_U_0[m,0]+=request[n,m]*weight[n,SELF]
                        else:
                            cache_U_0[m,0]+=request[n,m]*weight[n,D2D]
                    else:
                        cache_U_0[m, 0] += request[n, m] * weight[n, BASE]

            cache_U[:,0] -= cache_U_0[:,0]

            sort_cache_U=cache_U[np.lexsort(-cache_U[:,::-1].T)]
            cache[index,:]=np.zeros(M)
            for ca in range(S):

                add=sort_cache_U[ca,1]
                cache[index,int(add)]=1
                #print('用户：',index,'  存储下标:',add)


        hit_nums=np.zeros((N,M))
        U_self=0
        U_D2D = 0
        #print(cache.sum()==N*S)
        for n in range(N):
            for n_nearby in range(N):
                if communication[n,n_nearby]==0:
                    continue
                hit_nums[n,:]+=cache[n_nearby,:]
                #print(cache[n_nearby].sum())
            #print(hit_nums[n].sum())
        #print(hit_nums.sum())
        for n in range(N):
            for m in range(M):
                if hit_nums[n,m]>=1:
                    if cache[n,m]==1:
                        U_self+=request[n,m]*weight[n,SELF]
                    else:
                        U_D2D+=request[n,m]*weight[n,D2D]
                else:
                    U_BASE+=request[n,m]*weight[n,BASE]

        print(hit_nums.sum())
        print((U_D2D+U_self+U_BASE)/N)
        print(U_D2D/N,U_self/N,U_BASE/N)
        ans_U[i]=(U_D2D+U_self+U_BASE)/N

    # 表示总的效能
    U_P = 0
    # K表示循环论数 一定是 N的 K次方  一般里说  k=5就行了，不会比这个更高的
    K = 5
    # 把每一轮循环的结果 存起来
    ans_U_P = np.zeros(K)
    U_self_P = 0
    U_D2D_P = 0
    U_BASE_P = 0


    for i in range(K):
        random_user_set = np.arange(0,N,1)
        np.random.shuffle(random_user_set)
        print('这是第的',i*N,'轮迭代')
        #挑选一个用户 index 来是的他的缓存空间存储的对象可以使得 U全局 最大化
        for index in random_user_set:
            cache_U = np.zeros((M,2)) #用户 index 存入文件m 需要一个 表记录
            cache_U_0 = np.zeros((M,2))
            #cache_tmp矩阵表示临时表， index行全1
            cache_tmp =cache.copy()
            cache_tmp[index,:]=np.ones(M)
            cache_tmp_0 = cache.copy()
            cache_tmp_0[index,:]=np.zeros(M)
            #
            hit_nums = np.zeros((N,M))
            hit_nums_0 = np.zeros((N,M))
            for n in range(N):
                if communication[index,n]==0: #过滤所有的不能于index 通信的n
                    continue
                for n_nearby in range(N):
                    if communication[n,n_nearby]==0:
                        continue
                    hit_nums[n,:]+=cache_tmp[n_nearby,:] #得到n所有的hit数量
            for m in range(M):
                cache_U[m,1]=m
                for n in range(N):
                    if communication[index,n]==0: #过滤所有的不能于index 通信的n
                        continue
                    if hit_nums[n,m]>=1:
                        if cache_tmp[n,m]==1:
                            cache_U[m,0]+=request[n,m]*weight_P[n,SELF]
                        else:
                            cache_U[m,0]+=request[n,m]*weight_P[n,D2D]
                    else:
                        cache_U[m,0]+=request[n,m]*weight_P[n,BASE]

            for n in range(N):
                if communication[index,n]==0: #过滤所有的不能于index 通信的n
                    continue
                for n_nearby in range(N):
                    if communication[n,n_nearby]==0:
                        continue
                    hit_nums_0[n,:]+=cache_tmp_0[n_nearby,:] #得到n所有的hit数量
            for m in range(M):
                cache_U_0[m,1]=m
                for n in range(N):
                    if communication[index,n]==0: #过滤所有的不能于index 通信的n
                        continue
                    if hit_nums_0[n,m]>=1:
                        if cache_tmp_0[n,m]==1:
                            cache_U_0[m,0]+=request[n,m]*weight_P[n,SELF]
                        else:
                            cache_U_0[m,0]+=request[n,m]*weight_P[n,D2D]
                    else:
                        cache_U_0[m, 0] += request[n, m] * weight_P[n, BASE]

            cache_U[:,0] -= cache_U_0[:,0]

            sort_cache_U=cache_U[np.lexsort(-cache_U[:,::-1].T)]
            cache[index,:]=np.zeros(M)
            for ca in range(S):

                add=sort_cache_U[ca,1]
                cache[index,int(add)]=1
                #print('用户：',index,'  存储下标:',add)


        hit_nums=np.zeros((N,M))
        U_self_P=0
        U_D2D_P = 0
        U_BASE_P=0
        #print(cache.sum()==N*S)
        for n in range(N):
            for n_nearby in range(N):
                if communication[n,n_nearby]==0:
                    continue
                hit_nums[n,:]+=cache[n_nearby,:]
                #print(cache[n_nearby].sum())
            #print(hit_nums[n].sum())
        #print(hit_nums.sum())
        for n in range(N):
            for m in range(M):
                if hit_nums[n,m]>=1:
                    if cache[n,m]==1:
                        U_self_P+=request[n,m]*weight[n,SELF]
                    else:
                        U_D2D_P+=request[n,m]*weight[n,D2D]
                else:
                    U_BASE_P+=request[n,m]*weight[n,BASE]

        print(hit_nums.sum())
        print((U_D2D_P+U_self+U_BASE)/N)
        print(U_D2D_P/N,U_self_P/N,U_BASE_P/N)
        ans_U_P[i]=(U_D2D_P+U_self+U_BASE)/N



    #%%
    hit_nums.sum()
    #%% md

    #%% md

    #%%
    #self的话，每次用户只会存自己请求概率最高的S个文件
    self_cache =  np.zeros((N,M))
    for i in range(N):
        request_i=np.zeros((M,2))
        request_i[:,0]=request[i,:].copy().T
        request_i[:,1]=np.arange(0,M,1)
        sort_request_i=request_i[np.lexsort(-request_i[:,::-1].T)]
        for j in range(S):
            self_cache[i,int(sort_request_i[j,1])]=1



    #%%
    hit_nums=np.zeros((N,M))
    U_self_self=0
    U_self_D2D=0
    U_self_BASE= 0
    #print(cache.sum()==N*S)
    for n in range(N):
        for n_nearby in range(N):
            if communication[n,n_nearby]==0:
                continue
            hit_nums[n,:]+=self_cache[n_nearby,:]
            #print(cache[n_nearby].sum())
        #print(hit_nums[n].sum())
    #print(hit_nums.sum())
    for n in range(N):
        for m in range(M):
            if hit_nums[n,m]>=1:
                if self_cache[n,m]==1:
                    U_self_self+=request[n,m]*weight[n,SELF]
                else:
                    U_self_D2D+=request[n,m]*weight[n,D2D]
            else:
                U_self_BASE+=request[n,m]*weight[n,BASE]
    print((U_self_self+U_self_D2D+U_self_BASE)/N)

    #%% md

    #%%
    #global仍然是先处理request
    #这里的global不只是基站内的所有的用户的喜欢，而是全局 这个数据集的总偏好

    BS_rate = np.zeros(M)
    for i in range(M):
        BS_rate[i]= rate[:,i].sum()/N

    BS_request = np.zeros(M)
    for i in range(M):
        BS_request[i]=BS_rate[i]/BS_rate.sum()

    world_rate = 5*np.random.random(M)
    world_request = np.zeros(M)
    for i in range(M):
        world_request[i]=world_rate[i]/world_rate.sum()

    # global分为两种


    #%% md

    #%%
    BS_cache = np.zeros((N,M))
    #%%
    for i in range(S):
        tmp=np.arange(0,M,1)

        np.random.shuffle(tmp)
        for j in range(N):

            BS_cache[j,np.random.randint(0,len(tmp))]=1

    hit_nums=np.zeros((N,M))
    U_BS_global_D2D=0
    U_BS_global_self=0
    U_BS_global_BASE=0
    #print(cache.sum()==N*S)
    for n in range(N):
        for n_nearby in range(N):
            if communication[n,n_nearby]==0:
                continue
            hit_nums[n,:]+=BS_cache[n_nearby,:]
            #print(cache[n_nearby].sum())
        #print(hit_nums[n].sum())
    #print(hit_nums.sum())
    for n in range(N):
        for m in range(M):
            if hit_nums[n,m]>=1:
                if BS_cache[n,m]==1:
                    U_BS_global_self+=request[n,m]*weight[n,SELF]
                else:
                    U_BS_global_D2D+=request[n,m]*weight[n,D2D]
            else:
                U_BS_global_BASE+=request[n,m]*weight[n,BASE]

    print((U_BS_global_self+U_BS_global_D2D+U_BS_global_BASE)/N)
    #%% md

    #%%

    lr = 1/(N*S)
    value = (R_D2D**2)/(R_BS**2)
    Alternative = np.zeros(S*N)


    def d2d_gd(capacity):
        k = capacity*N
        results = np.zeros(M)
        for i in range(k):
            max=0
            index=0
            for j in range(len(results)):
                if d2d(world_request[j],results[j]+lr)-d2d(world_request[j],results[j])>max:
                    max =d2d(world_request[j],results[j]+lr)-d2d(world_request[j],results[j])
                    index=j

            Alternative[i]=index
            results[index]+=lr
        return results

    def d2d(a,x):
        return a-a*np.exp(-1.0*value* N * x)+a*x*np.exp(-1.0*value* N * x)

    result_gd = d2d_gd(S)
    #%%
    Alternative
    #%%
    world_cache=np.zeros((N,M))
    for i in range(N):
        for j in range(S):
            tmp=np.random.randint(0,len(Alternative)-1)
            world_cache[i,int(Alternative[tmp])]=1

    hit_nums=np.zeros((N,M))
    U_world_global_D2D=0
    U_world_global_self=0
    U_world_global_BASE=0
    #print(cache.sum()==N*S)
    for n in range(N):
        for n_nearby in range(N):
            if communication[n,n_nearby]==0:
                continue
            hit_nums[n,:]+=world_cache[n_nearby,:]
            #print(cache[n_nearby].sum())
        #print(hit_nums[n].sum())
    #print(hit_nums.sum())
    for n in range(N):
        for m in range(M):
            if hit_nums[n,m]>=1:
                if world_cache[n,m]==1:
                    U_world_global_self+=request[n,m]*weight[n,SELF]
                else:
                    U_world_global_D2D+=request[n,m]*weight[n,D2D]
            else:
                U_world_global_BASE+=request[n,m]*weight[n,BASE]

    print((U_world_global_self+U_world_global_D2D)/N)
    #%%
    print('个性化：',(U_self+U_D2D)/N)
    print('self：',(U_self_D2D+U_self_self)/N)
    print('BS global：',(U_BS_global_self+U_BS_global_D2D)/N)
    print('World global：',(U_world_global_self+U_world_global_D2D)/N)
    print('基于命中概率最大',(U_D2D_P+U_self_P+U_BASE_P)/N)
    y_1[variable] = (U_self+U_D2D+U_self_BASE)/N
    y_2[variable] =(U_self_D2D+U_self_self+U_self_BASE)/N
    y_3[variable] = (U_BS_global_self+U_BS_global_D2D+U_BS_global_BASE)/N
    y_4[variable] = (U_world_global_self+U_world_global_D2D+U_world_global_BASE)/N
    y_5[variable] = (U_D2D_P+U_self_P+U_BASE_P)/N


plt.title('x-D2D communicate distance , y-TH')
plt.plot(R_D2D_variable,y_1)
plt.plot(R_D2D_variable,y_2)
plt.plot(R_D2D_variable,y_3)
plt.plot(R_D2D_variable,y_4)
plt.plot(R_D2D_variable,y_5)
plt.show()


print(R_D2D_variable)
print(y_1)
print(y_2)
print(y_3)
print(y_4)
print(y_5)