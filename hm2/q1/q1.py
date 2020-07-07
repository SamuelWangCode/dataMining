#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random
import sklearn
from pandas import DataFrame
import math
import matplotlib.pyplot as plt
import datetime


# In[3]:


# 读取原始数据
row_data = pd.read_csv("F:/CourseData/数据挖掘/datamining20/trade_new.csv")
df = row_data[["sldatime", "pluno", "bndno", "qty"]]
df.loc[:,"kind1"] = (df.loc[:,"pluno"]/1000000).astype(int)
df.loc[:,"kind2"] = (df.loc[:,"pluno"]/100000).astype(int)
df.loc[:,"kind3"] = (df.loc[:,"pluno"]/10000).astype(int)
df.loc[:,"kind4"] = (df.loc[:,"pluno"]/1000).astype(int)
df


# In[5]:


arr = df.index[np.where(np.isnan(df.bndno))[0]]
for i in arr:
    df.loc[i,"bndno"] = df.loc[i,"pluno"]
df


# In[6]:


df["sldatime"].sort_values()


# In[7]:


# 规范化购买时间
df.loc[:,"sldatime"] = df.loc[:,"sldatime"].apply(lambda x:x[:10]).tolist()
df


# In[8]:


# 判断一天是第几个星期，第几个月
def judgeDate(a):
    delta = (datetime.datetime(int(a[0:4]),int(a[5:7]),int(a[8:10])) - datetime.datetime(2016,2,1)).days
    week = int(delta / 7)
    month = int(a[5:7])
    return (week, month)


# In[9]:


for i in range(df.shape[0]):
    df.loc[i,"week"] = judgeDate(df.loc[i,"sldatime"])[0]
    df.loc[i,"month"] = judgeDate(df.loc[i,"sldatime"])[1]
df


# In[16]:


# 按日期对商品购买记录排序
df = df.sort_values(by='sldatime')
df1 = df.groupby(["sldatime","pluno","bndno","kind1","kind2","kind3","kind4","week","month"]).sum()
df1 = df1.reset_index()


# In[62]:


# 提取需要的商品，舍弃其他的
a = df1.loc[df1["kind1"]== 22]
b = df1.loc[df1["kind1"]== 23]
c = df1.loc[df1["kind1"]== 25]
d = df1.loc[df1["kind1"]== 27]
e = df1.loc[df1["kind4"]== 15000]
df1 = a.append(b).append(c).append(d).append(e)
df1 = df1.reset_index(drop=True)
df1


# In[63]:


# 商品总表
plu_list = df1["pluno"].sort_values().drop_duplicates()
plu_list = plu_list.reset_index(drop="true")
plu_list


# In[64]:


# 品牌总表
bnd_list = df1["bndno"].sort_values().drop_duplicates()
bnd_list = bnd_list.reset_index(drop="true")
bnd_list


# In[65]:


# 一级品类总表
kind1_list = df1["kind1"].sort_values().drop_duplicates()
kind1_list = kind1_list.reset_index(drop="true")
kind1_list


# In[66]:


# 二级品类总表
kind2_list = df1["kind2"].sort_values().drop_duplicates()
kind2_list = kind2_list.reset_index(drop="true")
kind2_list


# In[67]:


# 三级品类总表
kind3_list = df1["kind3"].sort_values().drop_duplicates()
kind3_list = kind3_list.reset_index(drop="true")
kind3_list


# In[68]:


# 四级品类总表
kind4_list = df1["kind4"].sort_values().drop_duplicates()
kind4_list = kind4_list.reset_index(drop="true")
kind4_list


# In[69]:


# 时间总表
sldatime_list = []
i = datetime.datetime(2016,2,1)
while i <= datetime.datetime(2016,7,31):
    sldatime_list.append(i.strftime("%Y-%m-%d"))
    i = i + datetime.timedelta(1)
sldatime_list.__len__()


# In[70]:


# 周数和月数
week_list = df1["week"].sort_values().drop_duplicates().reset_index(drop="true").to_list()
month_list = df1["month"].sort_values().drop_duplicates().reset_index(drop="true").to_list()


# In[71]:


# 时间、商品编号对应的时间序列
df11 = pd.DataFrame([],columns=plu_list.to_list(),index=sldatime_list)


# In[72]:


for i in range(df1.shape[0]):
    df11.loc[df1.loc[i,"sldatime"],df1.loc[i,"pluno"]]=df1.loc[i,"qty"]
df11 = df11.fillna(0)
df11


# In[73]:


df11.to_csv("sldatime_pluno.csv")


# In[74]:


# 周、商品编号对应的时间序列
df12 = pd.DataFrame([],columns=plu_list.to_list(),index=week_list)
for i in range(df1.shape[0]):
    df12.loc[df1.loc[i,"week"],df1.loc[i,"pluno"]]=df1.loc[i,"qty"]
df12 = df12.fillna(0)
df12


# In[75]:


df12.to_csv("week_pluno.csv")


# In[76]:


# 月、商品编号对应的时间序列
df13 = pd.DataFrame([],columns=plu_list.to_list(),index=month_list)
for i in range(df1.shape[0]):
    df13.loc[df1.loc[i,"month"],df1.loc[i,"pluno"]]=df1.loc[i,"qty"]
df13 = df13.fillna(0)
df13


# In[77]:


df13.to_csv("month_pluno.csv")


# In[78]:


# 时间、品牌号对应的时间序列
df21 = pd.DataFrame([],columns=bnd_list.to_list(),index=sldatime_list)
for i in range(df1.shape[0]):
    df21.loc[df1.loc[i,"sldatime"],df1.loc[i,"bndno"]]=df1.loc[i,"qty"]
df21 = df21.fillna(0)
df21


# In[79]:


df21.to_csv("sldatime_bnd.csv")


# In[80]:


# 周、品牌号对应的时间序列
df22 = pd.DataFrame([],columns=bnd_list.to_list(),index=week_list)
for i in range(df1.shape[0]):
    df22.loc[df1.loc[i,"week"],df1.loc[i,"bndno"]]=df1.loc[i,"qty"]
df22 = df22.fillna(0)
df22


# In[81]:


df22.to_csv("week_bnd.csv")


# In[82]:


# 月、品牌号对应的时间序列
df23 = pd.DataFrame([],columns=bnd_list.to_list(),index=month_list)
for i in range(df1.shape[0]):
    df23.loc[df1.loc[i,"month"],df1.loc[i,"bndno"]]=df1.loc[i,"qty"]
df23 = df23.fillna(0)
df23


# In[83]:


df23.to_csv("month_bnd.csv")


# In[84]:


# 时间、一级品类 二级品类 三级品类 四级品类对应的时间序列
df31 = pd.DataFrame([],columns=kind1_list.to_list(),index=sldatime_list)
for i in range(df1.shape[0]):
    df31.loc[df1.loc[i,"sldatime"],df1.loc[i,"kind1"]]=df1.loc[i,"qty"]
df31 = df31.fillna(0)
df31.to_csv("sldatime_kind1.csv")

df41 = pd.DataFrame([],columns=kind2_list.to_list(),index=sldatime_list)
for i in range(df1.shape[0]):
    df41.loc[df1.loc[i,"sldatime"],df1.loc[i,"kind2"]]=df1.loc[i,"qty"]
df41 = df41.fillna(0)
df41.to_csv("sldatime_kind2.csv")

df51 = pd.DataFrame([],columns=kind3_list.to_list(),index=sldatime_list)
for i in range(df1.shape[0]):
    df51.loc[df1.loc[i,"sldatime"],df1.loc[i,"kind3"]]=df1.loc[i,"qty"]
df51 = df51.fillna(0)
df51.to_csv("sldatime_kind3.csv")

df61 = pd.DataFrame([],columns=kind4_list.to_list(),index=sldatime_list)
for i in range(df1.shape[0]):
    df61.loc[df1.loc[i,"sldatime"],df1.loc[i,"kind4"]]=df1.loc[i,"qty"]
df61 = df61.fillna(0)
df61.to_csv("sldatime_kind4.csv")

# 周、一级品类 二级品类 三级品类 四级品类对应的时间序列
df32 = pd.DataFrame([],columns=kind1_list.to_list(),index=week_list)
for i in range(df1.shape[0]):
    df32.loc[df1.loc[i,"week"],df1.loc[i,"kind1"]]=df1.loc[i,"qty"]
df32 = df32.fillna(0)
df32.to_csv("week_kind1.csv")

df42 = pd.DataFrame([],columns=kind2_list.to_list(),index=week_list)
for i in range(df1.shape[0]):
    df42.loc[df1.loc[i,"week"],df1.loc[i,"kind2"]]=df1.loc[i,"qty"]
df42 = df42.fillna(0)
df42.to_csv("week_kind2.csv")

df52 = pd.DataFrame([],columns=kind3_list.to_list(),index=week_list)
for i in range(df1.shape[0]):
    df52.loc[df1.loc[i,"week"],df1.loc[i,"kind3"]]=df1.loc[i,"qty"]
df52 = df52.fillna(0)
df52.to_csv("week_kind3.csv")

df62 = pd.DataFrame([],columns=kind4_list.to_list(),index=week_list)
for i in range(df1.shape[0]):
    df62.loc[df1.loc[i,"week"],df1.loc[i,"kind4"]]=df1.loc[i,"qty"]
df62 = df62.fillna(0)
df62.to_csv("week_kind4.csv")

# 月、一级品类 二级品类 三级品类 四级品类对应的时间序列
df33 = pd.DataFrame([],columns=kind1_list.to_list(),index=month_list)
for i in range(df1.shape[0]):
    df33.loc[df1.loc[i,"month"],df1.loc[i,"kind1"]]=df1.loc[i,"qty"]
df33 = df33.fillna(0)
df33.to_csv("month_kind1.csv")

df43 = pd.DataFrame([],columns=kind2_list.to_list(),index=month_list)
for i in range(df1.shape[0]):
    df43.loc[df1.loc[i,"month"],df1.loc[i,"kind2"]]=df1.loc[i,"qty"]
df43 = df43.fillna(0)
df43.to_csv("month_kind2.csv")

df53 = pd.DataFrame([],columns=kind3_list.to_list(),index=month_list)
for i in range(df1.shape[0]):
    df53.loc[df1.loc[i,"month"],df1.loc[i,"kind3"]]=df1.loc[i,"qty"]
df53 = df53.fillna(0)
df53.to_csv("month_kind3.csv")

df63 = pd.DataFrame([],columns=kind4_list.to_list(),index=month_list)
for i in range(df1.shape[0]):
    df63.loc[df1.loc[i,"month"],df1.loc[i,"kind4"]]=df1.loc[i,"qty"]
df63 = df63.fillna(0)
df63.to_csv("month_kind4.csv")


# In[49]:


df31


# In[51]:


df32


# In[52]:


df33


# In[53]:


df41


# In[54]:


df42


# In[55]:


df43


# In[56]:


df51


# In[57]:


df52


# In[58]:


df53


# In[59]:


df61


# In[60]:


df62


# In[61]:


df63


# In[ ]:




