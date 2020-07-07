#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
df1 = row_data[["vipno", "pluno", "amt"]]
df2 = row_data[["vipno", "pluno", "amt"]]
df3 = row_data[["vipno", "pluno", "amt"]]
df4 = row_data[["vipno", "pluno", "amt"]]


# In[8]:


# keyong 6227002180901670266 6227002180921663895 781924 13325038116
# 把pluno列取为第一、二、三、四级商品编号
df1["pluno"] = (df1["pluno"]/1000000).astype(int)
df2["pluno"] = (df2["pluno"]/100000).astype(int)
df3["pluno"] = (df3["pluno"]/10000).astype(int)
df4["pluno"] = (df4["pluno"]/1000).astype(int)
print(df1)


# **接下来为聚类数据做准备，先把所有的vip和plu列表整出来，然后这样就可以让一个用户占一行，然后列是每一个商品，也就是我们要进行聚类的维度**

# In[9]:


df1 = df1.sort_values(by=["vipno"])
vipno_series = df1["vipno"].drop_duplicates()
vipno_series = vipno_series.reset_index(drop=True)
print(vipno_series)
df1 = df1.sort_values(by=["pluno"])
pluno_series1 = df1["pluno"].drop_duplicates()
pluno_series1 = pluno_series1.reset_index(drop=True)
print(pluno_series1)
df2 = df2.sort_values(by=["pluno"])
pluno_series2 = df2["pluno"].drop_duplicates()
pluno_series2 = pluno_series2.reset_index(drop=True)
print(pluno_series2)
df3 = df3.sort_values(by=["pluno"])
pluno_series3 = df3["pluno"].drop_duplicates()
pluno_series3 = pluno_series3.reset_index(drop=True)
print(pluno_series3)
df4 = df4.sort_values(by=["pluno"])
pluno_series4 = df4["pluno"].drop_duplicates()
pluno_series4 = pluno_series4.reset_index(drop=True)
print(pluno_series4)
# 分组，求和
group_data1 = df1.groupby(["vipno","pluno"])["amt"].sum()
print(group_data1)
group_data2 = df2.groupby(["vipno","pluno"])["amt"].sum()
print(group_data2)
group_data3 = df3.groupby(["vipno","pluno"])["amt"].sum()
print(group_data3)
group_data4 = df4.groupby(["vipno","pluno"])["amt"].sum()
print(group_data4)
#记录：用户486个，plu一级18个，plu二级94个，plu三级329个，plu四级979个


# **这里是进行jaccard距离计算的函数，因为最后传入是两个数组，我们要取交集，除以并集**

# In[10]:


#记录：用户486个，plu一级18个，plu二级94个，plu三级329个，plu四级979个
def jaccard_dist(a, b):
    fenzi = 0
    fenmu = 0
    for i in range(0,18):
        fenzi += min(a[i],b[i])
        fenmu += max(a[i],b[i])
    sim1 = fenzi/fenmu
    fenzi = 0
    fenmu = 0
    for i in range(18,18+94):
        fenzi += min(a[i],b[i])
        fenmu += max(a[i],b[i])
    sim2 = fenzi/fenmu
    fenzi = 0
    fenmu = 0
    for i in range(18+94,18+94+329):
        fenzi += min(a[i],b[i])
        fenmu += max(a[i],b[i])
    sim3 = fenzi/fenmu
    fenzi = 0
    fenmu = 0
    for i in range(18+94+329,18+94+329+979):
        fenzi += min(a[i],b[i])
        fenmu += max(a[i],b[i])
    sim4 = fenzi/fenmu
    return 1-(sim1+sim2+sim3+sim4)/4


# In[35]:


distance = []
for i in range(len(data)):
    distance.append(jaccard_dist(data.values[0,:], data.values[i,:]))
plt.bar(range(len(data)), distance)


# **把每一个amt塞到横轴为用户，纵轴为商品的大表格里面**

# In[11]:


data1 = DataFrame(0, columns=pluno_series1, index=vipno_series)
# print(data)
for i in df1.index:
    vipno = df1['vipno'][i]
    pluno = df1['pluno'][i]
    amt = group_data1[vipno][pluno]
    if math.isnan(data1[pluno][vipno]):
        data1[pluno][vipno] = amt
    else:
        data1[pluno][vipno] += amt
data2 = DataFrame(0, columns=pluno_series2, index=vipno_series)
# print(data)
for i in df2.index:
    vipno = df2['vipno'][i]
    pluno = df2['pluno'][i]
    amt = group_data2[vipno][pluno]
    if math.isnan(data2[pluno][vipno]):
        data2[pluno][vipno] = amt
    else:
        data2[pluno][vipno] += amt
data3 = DataFrame(0, columns=pluno_series3, index=vipno_series)
# print(data)
for i in df1.index:
    vipno = df3['vipno'][i]
    pluno = df3['pluno'][i]
    amt = group_data3[vipno][pluno]
    if math.isnan(data3[pluno][vipno]):
        data3[pluno][vipno] = amt
    else:
        data3[pluno][vipno] += amt
data4 = DataFrame(0, columns=pluno_series4, index=vipno_series)
# print(data)
for i in df4.index:
    vipno = df4['vipno'][i]
    pluno = df4['pluno'][i]
    amt = group_data4[vipno][pluno]
    if math.isnan(data4[pluno][vipno]):
        data4[pluno][vipno] = amt
    else:
        data4[pluno][vipno] += amt


# In[12]:


data = pd.concat([data1, data2, data3, data4],axis=1)
print(data)


# In[13]:


#记录：用户486个，plu一级18个，plu二级94个，plu三级329个，plu四级979个
data.values[1,0:18]


# **下面是K-Means算法，由于距离计算函数不是欧式距离，所以要自己改成jaccard距离**

# In[21]:


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


# In[15]:


def CP(label, k, centroids, dataSet):
    cpnum = 0
    for i in range(k):
        distance = 0
        num = 0
        for j in range((len(label))):
            if label[j] == i:
                distance += jaccard_dist(dataSet.values[j,:],centroids[i,:])
                num += 1
        cpnum += distance/num
    return cpnum/k


# In[16]:


def getCentroid(dataSet):
    div = len(dataSet)
    return sum(dataSet)/div


# In[26]:


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


# In[18]:


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


# In[27]:


silhouette_score_array = []
for i in range(2,40):#从K为2到K为39，尝试一下
    silhouette_score_array.append(kmeans(data, i))


# In[28]:


print(silhouette_score_array)


# In[30]:


scdf = DataFrame(silhouette_score_array,index=range(2,40))
scdf


# In[31]:


print("SC指数变化趋势")
plt.plot(scdf[0])


# In[32]:


print("CP指数变化趋势")
plt.plot(scdf[1])


# In[33]:


test_sc_cp = kmeans(data, 200)


# In[34]:


test_sc_cp = kmeans(data, 300)

