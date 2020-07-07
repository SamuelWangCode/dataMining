#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pandas import DataFrame
import math
import matplotlib.pyplot as plt

# 读入数据所序列
row_data = pd.read_csv("F:/CourseData/数据挖掘/datamining20/trade_new.csv")
df = row_data[["vipno", "pluno", "amt"]]
print(df)


# In[36]:


# keyong 6227002180901670266 6227002180921663895 781924 13325038116
# 把pluno列取为第四级商品编号
df["pluno"] = (df["pluno"]/1000).astype(int)
print(df)


# **接下来为聚类数据做准备，先把所有的vip和plu列表整出来，然后这样就可以让一个用户占一行，然后列是每一个商品，也就是我们要进行聚类的维度**

# In[37]:


df = df.sort_values(by=["vipno"])
vipno_series = df["vipno"].drop_duplicates()
vipno_series = vipno_series.reset_index(drop=True)
print(vipno_series)
df = df.sort_values(by=["pluno"])
pluno_series = df["pluno"].drop_duplicates()
pluno_series = pluno_series.reset_index(drop=True)
print(pluno_series)
# 分组，求和
group_data = df.groupby(["vipno","pluno"])["amt"].sum()
print(group_data)


# **这里是进行jaccard距离计算的函数，因为最后传入是两个数组，我们要取交集，除以并集**

# In[38]:


# jaccard系数计算
def jaccard_sim(a, b):
    list=[]
    fenzi = 0
    fenmu = 0
    for i in group_data[a].index:
        if i in group_data[b].index:
            list.append(i)
    for i in group_data[a].index:
        if i in list:
            fenzi += min(group_data[a][i], group_data[b][i])
            fenmu += max(group_data[a][i], group_data[b][i])
        else:
            fenmu += group_data[a][i]
    for i in group_data[b].index:
        if i not in list:
            fenmu += group_data[b][i]
    return fenzi/fenmu


def jaccard_dist(a, b):
    fenzi = 0
    fenmu = 0
    for i in range(a.size):
        fenzi += min(a[i],b[i])
        fenmu += max(a[i],b[i])
    return 1-(fenzi/fenmu)


# In[92]:


distance = []
for i in range(len(data)):
    distance.append(jaccard_dist(data.values[0,:], data.values[i,:]))
plt.bar(range(len(data)), distance)


# **把每一个amt塞到横轴为用户，纵轴为商品的大表格里面**

# In[41]:


data = DataFrame(0, columns=pluno_series, index=vipno_series)
# print(data)
for i in df.index:
    vipno = df['vipno'][i]
    pluno = df['pluno'][i]
    amt = group_data[vipno][pluno]
    if math.isnan(data[pluno][vipno]):
        data[pluno][vipno] = amt
    else:
        data[pluno][vipno] += amt


# In[42]:


print(data)


# **下面是K-Means算法，由于距离计算函数不是欧式距离，所以要自己改成jaccard距离**

# In[65]:


def initCentroids(dataSet, k):#dataSet-数据点数组 k-设置的质心数
    #初始化质心 
    numSamples, dim = dataSet.shape#numSample-数据点个数 dim-数据点维数 
    #shape返回一个关于数组长宽的数组
    centroids = np.zeros((k, dim))#centroids-存放质心的数组
    index = random.sample(range(0, numSamples), k)#index-在零到数据点个数间的随机数
    print(index)
    for i in range(len(index)):
        centroids[i, :] = dataSet.values[index[i], :]
    #将随机质心存储入centroids
    return centroids


# In[44]:


def CP(label, k, centroids, dataSet):
    cpnum = 0
    for i in range(k):
        distance = 0
        num = 0
        for j in range(len(label)):
            if label[j] == i:
                distance += jaccard_dist(dataSet.values[j,:],centroids[i,:])
                num += 1
        cpnum += distance/num
    return cpnum/k


# In[45]:


def getCentroid(dataSet):
    div = len(dataSet)
    return sum(dataSet)/div


# In[70]:


def getSC(dataSet, label):
    sum_number = 0
    k = len(label)
    for i in range(k):
        ai = 0
        bi = 0
        anum = 0
        bnum = 0
        for j in range(k):
            if label[i]==label[j]:
                ai += jaccard_dist(dataSet.values[i,:],dataSet.values[j,:])
                anum += 1
            else:
                bi += jaccard_dist(dataSet.values[i,:],dataSet.values[j,:])
                bnum += 1
        ai = ai / anum
        bi = bi / bnum
        sum_number += (bi - ai) / max(ai, bi)     
    return sum_number / k


# In[59]:


def kmeans(dataSet, k):
    #k-means算法的核心函数
    numSamples = dataSet.shape[0]#数据点个数为数据点数组的行数
    label=np.zeros(dataSet.shape[0])
    clusterChanged = True#clusterChanged-表示是否需要重新分组的布尔值判定量
    
    centroids = initCentroids(dataSet, k)#初始化质心
    
    while clusterChanged:#需要重新分组时
        clusterChanged = False#重置判定量为假
        for i in range(numSamples):#遍历所有数据点
            minDist = 100000.0#minDist-最小的数据点与质心的距离
            minIndex = 0#minIndex-最小的链接地址
            for j in range(k):
                #计算每个数据点到哪个质心的距离最小，及记录是哪一个质心
                distance = jaccard_dist(centroids[j, :], dataSet.values[i, :])#distance-暂时存放数据点到质心的距离，这里是jaccard距离
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if label[i] != minIndex:#当该数据点所隶属的质心与最小链接地址不同时更新点中的数据
                clusterChanged = True#重置判定量为真
                label[i] = minIndex#该数据点的第二列变为一个数组
        for j in range(k):#由新的隶属关系中更新质心位置
            pointsInCluster = []
            for m in range(len(label)):
                if label[m]==j:
                    pointsInCluster.append(dataSet.values[m, :])
            centroids[j, :] = getCentroid(pointsInCluster)
        print(label)
    print("分类完成")
    #这里计算SC
    silhouette_score = getSC(data, label)
    #这里计算CP
    compactness_score = CP(label,k,centroids,dataSet)
    print("sc:" + str(silhouette_score))
    print("cp:" + str(compactness_score))
    return silhouette_score,compactness_score


# In[71]:


silhouette_score_array = []
for i in range(2,51):#从K为2到K为50，尝试一下
    silhouette_score_array.append(kmeans(data, i))


# In[72]:


print(silhouette_score_array)


# In[74]:


scdf = DataFrame(silhouette_score_array,index=range(2,51))
scdf


# In[75]:


print("SC指数变化趋势")
plt.plot(scdf[0])


# In[76]:


print("CP指数变化趋势")
plt.plot(scdf[1])


# In[77]:


test_sc_cp = kmeans(data, 100)


# In[78]:


test_sc_cp = kmeans(data, 200)


# In[79]:


test_sc_cp = kmeans(data, 300)


# In[80]:


test_sc_cp = kmeans(data, 400)

