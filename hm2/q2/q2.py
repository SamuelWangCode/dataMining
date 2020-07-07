#!/usr/bin/env python
# coding: utf-8

# In[256]:


import pandas as pd
import numpy as np
import random
import sklearn
from pandas import DataFrame
import math
import matplotlib.pyplot as plt
import datetime


# In[368]:


df = pd.read_csv("../q1/sum_table.csv")
df
# 提取需要的商品，舍弃其他的
a = df.loc[df["kind1"]== 22]
b = df.loc[df["kind1"]== 23]
c = df.loc[df["kind1"]== 25]
d = df.loc[df["kind1"]== 27]
e = df.loc[df["kind4"]== 15000]
df = a.append(b).append(c).append(d).append(e)
df = df.reset_index(drop=True)
df = df.sort_values(by=["pluno","sldatime"]).reset_index(drop=True)
df


# In[263]:


sldatime_list = []
i = datetime.datetime(2016,2,1)
while i <= datetime.datetime(2016,7,31):
    sldatime_list.append(i.strftime("%Y-%m-%d"))
    i = i + datetime.timedelta(1)
sldatime_list


# In[340]:


plu_list = df["pluno"].sort_values().drop_duplicates()
plu_list = plu_list.reset_index(drop="true")
plu_list


# In[369]:


df = df.drop(columns=["week","month"])


# In[379]:


dataframe = df.loc[df["pluno"]==plu_list[0]].reset_index(drop=True)
dataframe.kind1[0]


# In[380]:


for i in range(plu_list.__len__()):
    print(i)
    dataframe = df.loc[df["pluno"]==plu_list[i]].reset_index(drop=True)
    alist = dataframe.sldatime.to_list()
    for j in range(sldatime_list.__len__()):
        if sldatime_list[j] not in alist:
            df.loc[df.shape[0]] = {'sldatime':sldatime_list[j],'pluno':dataframe.pluno[0],'bndno':dataframe.bndno[0],'kind1':dataframe.kind1[0],'kind2':dataframe.kind2[0],'kind3':dataframe.kind3[0],'kind4':dataframe.kind4[0],'qty':0}
df


# In[381]:


df = df.sort_values(by=["pluno","sldatime"]).reset_index(drop=True)


# In[367]:


plu_list.__len__()*sldatime_list.__len__()


# In[342]:


df_plu = pd.read_csv("../q1/sldatime_pluno.csv", index_col=0)
df_plu


# In[343]:


df_bnd = pd.read_csv("../q1/sldatime_bnd.csv", index_col=0)
df_bnd


# In[344]:


df_kind1 = pd.read_csv("../q1/sldatime_kind1.csv", index_col=0)
df_kind1


# In[345]:


df_kind2 = pd.read_csv("../q1/sldatime_kind2.csv", index_col=0)
df_kind2


# In[346]:


df_kind3 = pd.read_csv("../q1/sldatime_kind3.csv", index_col=0)
df_kind3


# In[347]:


df_kind4 = pd.read_csv("../q1/sldatime_kind4.csv", index_col=0)
df_kind4


# In[295]:


ndf = pd.DataFrame([],columns=["pluno","bndno","kind1","kind2","kind3","kind4","sldatime","isWeekday","qty","qty1","qty2","qty3","qty4","qty5","qty6","qty7","bndqty1","bndqty2","bndqty3","bndqty4","bndqty5","bndqty6","bndqty7","kind1qty1","kind1qty2","kind1qty3","kind1qty4","kind1qty5","kind1qty6","kind1qty7","kind2qty1","kind2qty2","kind2qty3","kind2qty4","kind2qty5","kind2qty6","kind2qty7","kind3qty1","kind3qty2","kind3qty3","kind3qty4","kind3qty5","kind3qty6","kind3qty7","kind4qty1","kind4qty2","kind4qty3","kind4qty4","kind4qty5","kind4qty6","kind4qty7","week2AvgQty","week2MaxQty","week2MinQty","week3AvgQty","week3MaxQty","week3MinQty","week4AvgQty","week4MaxQty","week4MinQty","week2AvgBndQty","week2MaxBndQty","week2MinBndQty","week3AvgBndQty","week3MaxBndQty","week3MinBndQty","week4AvgBndQty","week4MaxBndQty","week4MinBndQty","week2AvgKind1Qty","week2MaxKind1Qty","week2MinKind1Qty","week3AvgKind1Qty","week3MaxKind1Qty","week3MinKind1Qty","week4AvgKind1Qty","week4MaxKind1Qty","week4MinKind1Qty","week2AvgKind2Qty","week2MaxKind2Qty","week2MinKind2Qty","week3AvgKind2Qty","week3MaxKind2Qty","week3MinKind2Qty","week4AvgKind2Qty","week4MaxKind2Qty","week4MinKind2Qty","week2AvgKind3Qty","week2MaxKind3Qty","week2MinKind3Qty","week3AvgKind3Qty","week3MaxKind3Qty","week3MinKind3Qty","week4AvgKind3Qty","week4MaxKind3Qty","week4MinKind3Qty","week2AvgKind4Qty","week2MaxKind4Qty","week2MinKind4Qty","week3AvgKind4Qty","week3MaxKind4Qty","week3MinKind4Qty","week4AvgKind4Qty","week4MaxKind4Qty","week4MinKind4Qty"])
ndf


# In[313]:


bnd_dict = df[["pluno","bndno"]].drop_duplicates().reset_index(drop=True).set_index("pluno").T.to_dict()


# In[317]:


bnd_dict.get(15000000)['bndno']


# In[325]:


(plu_list[0]/1000000).astype(int)


# In[329]:


df_bnd.loc[sldatime_list[0],str(bnd_dict.get(plu_list[0])['bndno'])]


# In[336]:


np.max([0.2,0.5])


# In[387]:


for i in range(df.shape[0]):
    df.loc[i, "isWeekday"] = not (df.index[i]%7==5 or df.index[i]%7==6)


# In[394]:


for i in range(df.shape[0]):
    df.loc[i,"isWeekday"] = int(df.loc[i,"isWeekday"])
df


# In[395]:


# 存档
sdf = df.copy(deep=True)


# In[397]:


feature1 = pd.DataFrame(columns=['sldatime','pluno','bndno','kind1','kind2','kind3','kind4','qty','isWeekday','qty1','qty2','qty3','qty4','qty5','qty6','qty7'])
feature1


# In[398]:


for i in plu_list:
    plu=df[df['pluno'].isin([i])]
    for day in range(1,8):
        d_sales = 'qty' + str(day)
        plu[d_sales] = plu['qty'].shift(day)
    feature1 = feature1.append(plu, ignore_index=True)
feature1 = feature1.fillna(0)
feature1


# In[399]:


feature1.to_csv("feature1.csv")


# In[420]:


s4 = pd.DataFrame(columns=['sldatime','pluno','bndno','kind1','kind2','kind3','kind4','qty','isWeekday','qty8','qty9','qty10','qty11','qty12','qty13','qty14','qty15','qty16','qty17','qty18','qty19','qty20','qty21','qty22','qty23','qty24','qty25','qty26','qty27','qty28'])


# In[422]:


for i in plu_list:
    print(i)
    plu=df[df['pluno'].isin([i])]
    for day in range(8,29):
        d_sales = 'qty' + str(day)
        plu[d_sales] = plu['qty'].shift(day)
    s4 = s4.append(plu, ignore_index=True)
s4 = s4.fillna(0)
s4 = s4.drop(['bndno','kind1','kind2','kind3','kind4','isWeekday'],axis = 1)
s4


# In[427]:


s4 = s4.drop_duplicates().reset_index(drop=True)


# In[429]:


feature4 = pd.DataFrame(columns=["sldatime","pluno","qty","week2AvgQty","week2MaxQty","week2MinQty","week3AvgQty","week3MaxQty","week3MinQty","week4AvgQty","week4MaxQty","week4MinQty"])


# In[431]:


for i in range(s4.shape[0]):
    feature4.loc[i] = [s4.loc[i,'sldatime'],s4.loc[i,'pluno'],s4.loc[i,"qty"],np.mean([s4.loc[i,"qty8"],s4.loc[i,"qty9"],s4.loc[i,"qty10"],s4.loc[i,"qty11"],s4.loc[i,"qty12"],s4.loc[i,"qty13"],s4.loc[i,"qty14"]]),np.max([s4.loc[i,"qty8"],s4.loc[i,"qty9"],s4.loc[i,"qty10"],s4.loc[i,"qty11"],s4.loc[i,"qty12"],s4.loc[i,"qty13"],s4.loc[i,"qty14"]]),np.min([s4.loc[i,"qty8"],s4.loc[i,"qty9"],s4.loc[i,"qty10"],s4.loc[i,"qty11"],s4.loc[i,"qty12"],s4.loc[i,"qty13"],s4.loc[i,"qty14"]]),np.mean([s4.loc[i,"qty15"],s4.loc[i,"qty16"],s4.loc[i,"qty17"],s4.loc[i,"qty18"],s4.loc[i,"qty19"],s4.loc[i,"qty20"],s4.loc[i,"qty21"]]),np.max([s4.loc[i,"qty15"],s4.loc[i,"qty16"],s4.loc[i,"qty17"],s4.loc[i,"qty18"],s4.loc[i,"qty19"],s4.loc[i,"qty20"],s4.loc[i,"qty21"]]),np.min([s4.loc[i,"qty15"],s4.loc[i,"qty16"],s4.loc[i,"qty17"],s4.loc[i,"qty18"],s4.loc[i,"qty19"],s4.loc[i,"qty20"],s4.loc[i,"qty21"]]),np.mean([s4.loc[i,"qty22"],s4.loc[i,"qty23"],s4.loc[i,"qty24"],s4.loc[i,"qty25"],s4.loc[i,"qty26"],s4.loc[i,"qty27"],s4.loc[i,"qty28"]]),np.max([s4.loc[i,"qty22"],s4.loc[i,"qty23"],s4.loc[i,"qty24"],s4.loc[i,"qty25"],s4.loc[i,"qty26"],s4.loc[i,"qty27"],s4.loc[i,"qty28"]]),np.min([s4.loc[i,"qty22"],s4.loc[i,"qty23"],s4.loc[i,"qty24"],s4.loc[i,"qty25"],s4.loc[i,"qty26"],s4.loc[i,"qty27"],s4.loc[i,"qty28"]])]
    print(i)
feature4


# In[433]:


feature4.to_csv("feature4.csv")


# In[439]:


feature2 = pd.DataFrame(columns=['sldatime','pluno','bndno','qty','bndqty1','bndqty2','bndqty3','bndqty4','bndqty5','bndqty6','bndqty7'])
feature2


# In[435]:


# feature1.groupby(["bndno","sldatime"]).sum().loc[15011,'2016-02-01'].qty
bndqty_df = df.drop(["pluno","kind1","kind2","kind3","kind4","isWeekday"],axis=1).groupby(["bndno","sldatime"],as_index=False).sum()
bndqty_df


# In[440]:


for i in plu_list:
    print(i)
    plu=df[df['pluno'].isin([i])]
    for day in range(1,8):
        d_sales = 'bndqty' + str(day)
        bnd = bndqty_df[bndqty_df['bndno']==bnd_dict.get(i)['bndno']]
        plu[d_sales] = bnd['qty'].shift(day)
    feature2 = feature2.append(plu, ignore_index=True)
feature2 = feature2.fillna(0)
feature2


# In[441]:


feature2 = feature2.drop(["isWeekday","kind1","kind2","kind3","kind4"],axis = 1)
feature2.to_csv("feature2.csv")


# In[443]:


s5 = pd.DataFrame(columns=['sldatime','pluno','bndno','kind1','kind2','kind3','kind4','qty','isWeekday','bndqty8','bndqty9','bndqty10','bndqty11','bndqty12','bndqty13','bndqty14','bndqty15','bndqty16','bndqty17','bndqty18','bndqty19','bndqty20','bndqty21','bndqty22','bndqty23','bndqty24','bndqty25','bndqty26','bndqty27','bndqty28'])


# In[444]:


for i in plu_list:
    print(i)
    plu=df[df['pluno'].isin([i])]
    for day in range(8,29):
        d_sales = 'bndqty' + str(day)
        bnd = bndqty_df[bndqty_df['bndno']==bnd_dict.get(i)['bndno']]
        plu[d_sales] = bnd['qty'].shift(day)
    s5 = s5.append(plu, ignore_index=True)
s5 = s5.fillna(0)
s5


# In[445]:


feature5 = pd.DataFrame(columns=["sldatime","pluno","qty","week2AvgBndQty","week2MaxBndQty","week2MinBndQty","week3AvgBndQty","week3MaxBndQty","week3MinBndQty","week4AvgBndQty","week4MaxBndQty","week4MinBndQty"])


# In[446]:


for i in range(s5.shape[0]):
    feature5.loc[i] = [s5.loc[i,'sldatime'],s5.loc[i,'pluno'],s5.loc[i,"qty"],np.mean([s5.loc[i,"bndqty8"],s5.loc[i,"bndqty9"],s5.loc[i,"bndqty10"],s5.loc[i,"bndqty11"],s5.loc[i,"bndqty12"],s5.loc[i,"bndqty13"],s5.loc[i,"bndqty14"]]),np.max([s5.loc[i,"bndqty8"],s5.loc[i,"bndqty9"],s5.loc[i,"bndqty10"],s5.loc[i,"bndqty11"],s5.loc[i,"bndqty12"],s5.loc[i,"bndqty13"],s5.loc[i,"bndqty14"]]),np.min([s5.loc[i,"bndqty8"],s5.loc[i,"bndqty9"],s5.loc[i,"bndqty10"],s5.loc[i,"bndqty11"],s5.loc[i,"bndqty12"],s5.loc[i,"bndqty13"],s5.loc[i,"bndqty14"]]),np.mean([s5.loc[i,"bndqty15"],s5.loc[i,"bndqty16"],s5.loc[i,"bndqty17"],s5.loc[i,"bndqty18"],s5.loc[i,"bndqty19"],s5.loc[i,"bndqty20"],s5.loc[i,"bndqty21"]]),np.max([s5.loc[i,"bndqty15"],s5.loc[i,"bndqty16"],s5.loc[i,"bndqty17"],s5.loc[i,"bndqty18"],s5.loc[i,"bndqty19"],s5.loc[i,"bndqty20"],s5.loc[i,"bndqty21"]]),np.min([s5.loc[i,"bndqty15"],s5.loc[i,"bndqty16"],s5.loc[i,"bndqty17"],s5.loc[i,"bndqty18"],s5.loc[i,"bndqty19"],s5.loc[i,"bndqty20"],s5.loc[i,"bndqty21"]]),np.mean([s5.loc[i,"bndqty22"],s5.loc[i,"bndqty23"],s5.loc[i,"bndqty24"],s5.loc[i,"bndqty25"],s5.loc[i,"bndqty26"],s5.loc[i,"bndqty27"],s5.loc[i,"bndqty28"]]),np.max([s5.loc[i,"bndqty22"],s5.loc[i,"bndqty23"],s5.loc[i,"bndqty24"],s5.loc[i,"bndqty25"],s5.loc[i,"bndqty26"],s5.loc[i,"bndqty27"],s5.loc[i,"bndqty28"]]),np.min([s5.loc[i,"bndqty22"],s5.loc[i,"bndqty23"],s5.loc[i,"bndqty24"],s5.loc[i,"bndqty25"],s5.loc[i,"bndqty26"],s5.loc[i,"bndqty27"],s5.loc[i,"bndqty28"]])]
    print(i)
feature5


# In[448]:


feature5.to_csv("feature5.csv")


# In[449]:


kind1_df = df.drop(["pluno","bndno","kind2","kind3","kind4","isWeekday"],axis=1).groupby(["kind1","sldatime"],as_index=False).sum()
kind2_df = df.drop(["pluno","bndno","kind1","kind3","kind4","isWeekday"],axis=1).groupby(["kind2","sldatime"],as_index=False).sum()
kind3_df = df.drop(["pluno","bndno","kind1","kind2","kind4","isWeekday"],axis=1).groupby(["kind3","sldatime"],as_index=False).sum()
kind4_df = df.drop(["pluno","bndno","kind2","kind3","kind1","isWeekday"],axis=1).groupby(["kind4","sldatime"],as_index=False).sum()


# In[450]:


kind1_df


# In[451]:


kind2_df


# In[452]:


kind3_df


# In[453]:


kind4_df


# In[454]:


feature3 = pd.DataFrame(columns=['sldatime','pluno','bndno','qty','kind1qty1','kind1qty2','kind1qty3','kind1qty4','kind1qty5','kind1qty6','kind1qty7','kind2qty1','kind2qty2','kind2qty3','kind2qty4','kind2qty5','kind2qty6','kind2qty7','kind3qty1','kind3qty2','kind3qty3','kind3qty4','kind3qty5','kind3qty6','kind3qty7','kind4qty1','kind4qty2','kind4qty3','kind4qty4','kind4qty5','kind4qty6','kind4qty7'])
feature3


# In[458]:


for i in plu_list:
    print(i)
    plu=df[df['pluno'].isin([i])]
    for day in range(1,8):
        d_sales = 'kind1qty' + str(day)
        kind1 = kind1_df[kind1_df['kind1']==int(i/1000000)]
        plu[d_sales] = kind1['qty'].shift(day)
    for day in range(1,8):
        d_sales = 'kind2qty' + str(day)
        kind2 = kind2_df[kind2_df['kind2']==int(i/100000)]
        plu[d_sales] = kind2['qty'].shift(day)
    for day in range(1,8):
        d_sales = 'kind3qty' + str(day)
        kind3 = kind3_df[kind3_df['kind3']==int(i/10000)]
        plu[d_sales] = kind3['qty'].shift(day)
    for day in range(1,8):
        d_sales = 'kind4qty' + str(day)
        kind4 = kind4_df[kind4_df['kind4']==int(i/1000)]
        plu[d_sales] = kind4['qty'].shift(day)
    feature3 = feature3.append(plu, ignore_index=True)
feature3 = feature3.fillna(0)
feature3


# In[459]:


feature3.to_csv("feature3.csv")


# In[460]:


s6 = pd.DataFrame(columns=['sldatime','pluno','bndno','kind1','kind2','kind3','kind4','qty','isWeekday','kind1qty8','kind1qty9','kind1qty10','kind1qty11','kind1qty12','kind1qty13','kind1qty14','kind1qty15','kind1qty16','kind1qty17','kind1qty18','kind1qty19','kind1qty20','kind1qty21','kind1qty22','kind1qty23','kind1qty24','kind1qty25','kind1qty26','kind1qty27','kind1qty28','kind2qty8','kind2qty9','kind2qty10','kind2qty11','kind2qty12','kind2qty13','kind2qty14','kind2qty15','kind2qty16','kind2qty17','kind2qty18','kind2qty19','kind2qty20','kind2qty21','kind2qty22','kind2qty23','kind2qty24','kind2qty25','kind2qty26','kind2qty27','kind2qty28','kind3qty8','kind3qty9','kind3qty10','kind3qty11','kind3qty12','kind3qty13','kind3qty14','kind3qty15','kind3qty16','kind3qty17','kind3qty18','kind3qty19','kind3qty20','kind3qty21','kind3qty22','kind3qty23','kind3qty24','kind3qty25','kind3qty26','kind3qty27','kind3qty28','kind4qty8','kind4qty9','kind4qty10','kind4qty11','kind4qty12','kind4qty13','kind4qty14','kind4qty15','kind4qty16','kind4qty17','kind4qty18','kind4qty19','kind4qty20','kind4qty21','kind4qty22','kind4qty23','kind4qty24','kind4qty25','kind4qty26','kind4qty27','kind4qty28'])


# In[461]:


for i in plu_list:
    print(i)
    plu=df[df['pluno'].isin([i])]
    for day in range(8,29):
        d_sales = 'kind1qty' + str(day)
        kind1 = kind1_df[kind1_df['kind1']==int(i/1000000)]
        plu[d_sales] = kind1['qty'].shift(day)
    for day in range(8,29):
        d_sales = 'kind2qty' + str(day)
        kind2 = kind2_df[kind2_df['kind2']==int(i/100000)]
        plu[d_sales] = kind2['qty'].shift(day)
    for day in range(8,29):
        d_sales = 'kind3qty' + str(day)
        kind3 = kind3_df[kind3_df['kind3']==int(i/10000)]
        plu[d_sales] = kind3['qty'].shift(day)
    for day in range(8,29):
        d_sales = 'kind4qty' + str(day)
        kind4 = kind4_df[kind4_df['kind4']==int(i/1000)]
        plu[d_sales] = kind4['qty'].shift(day)
    s6 = s6.append(plu, ignore_index=True)
s6 = s6.fillna(0)
s6


# In[465]:


feature6 = pd.DataFrame(columns=["sldatime","pluno","qty","week2AvgKind1Qty","week2MaxKind1Qty","week2MinKind1Qty","week3AvgKind1Qty","week3MaxKind1Qty","week3MinKind1Qty","week4AvgKind1Qty","week4MaxKind1Qty","week4MinKind1Qty","week2AvgKind2Qty","week2MaxKind2Qty","week2MinKind2Qty","week3AvgKind2Qty","week3MaxKind2Qty","week3MinKind2Qty","week4AvgKind2Qty","week4MaxKind2Qty","week4MinKind2Qty","week2AvgKind3Qty","week2MaxKind3Qty","week2MinKind3Qty","week3AvgKind3Qty","week3MaxKind3Qty","week3MinKind3Qty","week4AvgKind3Qty","week4MaxKind3Qty","week4MinKind3Qty","week2AvgKind4Qty","week2MaxKind4Qty","week2MinKind4Qty","week3AvgKind4Qty","week3MaxKind4Qty","week3MinKind4Qty","week4AvgKind4Qty","week4MaxKind4Qty","week4MinKind4Qty"])


# In[469]:


for i in range(s6.shape[0]):
    feature6.loc[i] = [s6.loc[i,'sldatime'],s6.loc[i,'pluno'],s6.loc[i,'qty'],np.mean([s6.loc[i,"kind1qty8"],s6.loc[i,"kind1qty9"],s6.loc[i,"kind1qty10"],s6.loc[i,"kind1qty11"],s6.loc[i,"kind1qty12"],s6.loc[i,"kind1qty13"],s6.loc[i,"kind1qty14"]]),np.max([s6.loc[i,"kind1qty8"],s6.loc[i,"kind1qty9"],s6.loc[i,"kind1qty10"],s6.loc[i,"kind1qty11"],s6.loc[i,"kind1qty12"],s6.loc[i,"kind1qty13"],s6.loc[i,"kind1qty14"]]),np.min([s6.loc[i,"kind1qty8"],s6.loc[i,"kind1qty9"],s6.loc[i,"kind1qty10"],s6.loc[i,"kind1qty11"],s6.loc[i,"kind1qty12"],s6.loc[i,"kind1qty13"],s6.loc[i,"kind1qty14"]]),np.mean([s6.loc[i,"kind1qty15"],s6.loc[i,"kind1qty16"],s6.loc[i,"kind1qty17"],s6.loc[i,"kind1qty18"],s6.loc[i,"kind1qty19"],s6.loc[i,"kind1qty20"],s6.loc[i,"kind1qty21"]]),np.max([s6.loc[i,"kind1qty15"],s6.loc[i,"kind1qty16"],s6.loc[i,"kind1qty17"],s6.loc[i,"kind1qty18"],s6.loc[i,"kind1qty19"],s6.loc[i,"kind1qty20"],s6.loc[i,"kind1qty21"]]),np.min([s6.loc[i,"kind1qty15"],s6.loc[i,"kind1qty16"],s6.loc[i,"kind1qty17"],s6.loc[i,"kind1qty18"],s6.loc[i,"kind1qty19"],s6.loc[i,"kind1qty20"],s6.loc[i,"kind1qty21"]]),np.mean([s6.loc[i,"kind1qty22"],s6.loc[i,"kind1qty23"],s6.loc[i,"kind1qty24"],s6.loc[i,"kind1qty25"],s6.loc[i,"kind1qty26"],s6.loc[i,"kind1qty27"],s6.loc[i,"kind1qty28"]]),np.max([s6.loc[i,"kind1qty22"],s6.loc[i,"kind1qty23"],s6.loc[i,"kind1qty24"],s6.loc[i,"kind1qty25"],s6.loc[i,"kind1qty26"],s6.loc[i,"kind1qty27"],s6.loc[i,"kind1qty28"]]),np.min([s6.loc[i,"kind1qty22"],s6.loc[i,"kind1qty23"],s6.loc[i,"kind1qty24"],s6.loc[i,"kind1qty25"],s6.loc[i,"kind1qty26"],s6.loc[i,"kind1qty27"],s6.loc[i,"kind1qty28"]]),np.mean([s6.loc[i,"kind2qty8"],s6.loc[i,"kind2qty9"],s6.loc[i,"kind2qty10"],s6.loc[i,"kind2qty11"],s6.loc[i,"kind2qty12"],s6.loc[i,"kind2qty13"],s6.loc[i,"kind2qty14"]]),np.max([s6.loc[i,"kind2qty8"],s6.loc[i,"kind2qty9"],s6.loc[i,"kind2qty10"],s6.loc[i,"kind2qty11"],s6.loc[i,"kind2qty12"],s6.loc[i,"kind2qty13"],s6.loc[i,"kind2qty14"]]),np.min([s6.loc[i,"kind2qty8"],s6.loc[i,"kind2qty9"],s6.loc[i,"kind2qty10"],s6.loc[i,"kind2qty11"],s6.loc[i,"kind2qty12"],s6.loc[i,"kind2qty13"],s6.loc[i,"kind2qty14"]]),np.mean([s6.loc[i,"kind2qty15"],s6.loc[i,"kind2qty16"],s6.loc[i,"kind2qty17"],s6.loc[i,"kind2qty18"],s6.loc[i,"kind2qty19"],s6.loc[i,"kind2qty20"],s6.loc[i,"kind2qty21"]]),np.max([s6.loc[i,"kind2qty15"],s6.loc[i,"kind2qty16"],s6.loc[i,"kind2qty17"],s6.loc[i,"kind2qty18"],s6.loc[i,"kind2qty19"],s6.loc[i,"kind2qty20"],s6.loc[i,"kind2qty21"]]),np.min([s6.loc[i,"kind2qty15"],s6.loc[i,"kind2qty16"],s6.loc[i,"kind2qty17"],s6.loc[i,"kind2qty18"],s6.loc[i,"kind2qty19"],s6.loc[i,"kind2qty20"],s6.loc[i,"kind2qty21"]]),np.mean([s6.loc[i,"kind2qty22"],s6.loc[i,"kind2qty23"],s6.loc[i,"kind2qty24"],s6.loc[i,"kind2qty25"],s6.loc[i,"kind2qty26"],s6.loc[i,"kind2qty27"],s6.loc[i,"kind2qty28"]]),np.max([s6.loc[i,"kind2qty22"],s6.loc[i,"kind2qty23"],s6.loc[i,"kind2qty24"],s6.loc[i,"kind2qty25"],s6.loc[i,"kind2qty26"],s6.loc[i,"kind2qty27"],s6.loc[i,"kind2qty28"]]),np.min([s6.loc[i,"kind2qty22"],s6.loc[i,"kind2qty23"],s6.loc[i,"kind2qty24"],s6.loc[i,"kind2qty25"],s6.loc[i,"kind2qty26"],s6.loc[i,"kind2qty27"],s6.loc[i,"kind2qty28"]]),np.mean([s6.loc[i,"kind3qty8"],s6.loc[i,"kind3qty9"],s6.loc[i,"kind3qty10"],s6.loc[i,"kind3qty11"],s6.loc[i,"kind3qty12"],s6.loc[i,"kind3qty13"],s6.loc[i,"kind3qty14"]]),np.max([s6.loc[i,"kind3qty8"],s6.loc[i,"kind3qty9"],s6.loc[i,"kind3qty10"],s6.loc[i,"kind3qty11"],s6.loc[i,"kind3qty12"],s6.loc[i,"kind3qty13"],s6.loc[i,"kind3qty14"]]),np.min([s6.loc[i,"kind3qty8"],s6.loc[i,"kind3qty9"],s6.loc[i,"kind3qty10"],s6.loc[i,"kind3qty11"],s6.loc[i,"kind3qty12"],s6.loc[i,"kind3qty13"],s6.loc[i,"kind3qty14"]]),np.mean([s6.loc[i,"kind3qty15"],s6.loc[i,"kind3qty16"],s6.loc[i,"kind3qty17"],s6.loc[i,"kind3qty18"],s6.loc[i,"kind3qty19"],s6.loc[i,"kind3qty20"],s6.loc[i,"kind3qty21"]]),np.max([s6.loc[i,"kind3qty15"],s6.loc[i,"kind3qty16"],s6.loc[i,"kind3qty17"],s6.loc[i,"kind3qty18"],s6.loc[i,"kind3qty19"],s6.loc[i,"kind3qty20"],s6.loc[i,"kind3qty21"]]),np.min([s6.loc[i,"kind3qty15"],s6.loc[i,"kind3qty16"],s6.loc[i,"kind3qty17"],s6.loc[i,"kind3qty18"],s6.loc[i,"kind3qty19"],s6.loc[i,"kind3qty20"],s6.loc[i,"kind3qty21"]]),np.mean([s6.loc[i,"kind3qty22"],s6.loc[i,"kind3qty23"],s6.loc[i,"kind3qty24"],s6.loc[i,"kind3qty25"],s6.loc[i,"kind3qty26"],s6.loc[i,"kind3qty27"],s6.loc[i,"kind3qty28"]]),np.max([s6.loc[i,"kind3qty22"],s6.loc[i,"kind3qty23"],s6.loc[i,"kind3qty24"],s6.loc[i,"kind3qty25"],s6.loc[i,"kind3qty26"],s6.loc[i,"kind3qty27"],s6.loc[i,"kind3qty28"]]),np.min([s6.loc[i,"kind3qty22"],s6.loc[i,"kind3qty23"],s6.loc[i,"kind3qty24"],s6.loc[i,"kind3qty25"],s6.loc[i,"kind3qty26"],s6.loc[i,"kind3qty27"],s6.loc[i,"kind3qty28"]]),np.mean([s6.loc[i,"kind4qty8"],s6.loc[i,"kind4qty9"],s6.loc[i,"kind4qty10"],s6.loc[i,"kind4qty11"],s6.loc[i,"kind4qty12"],s6.loc[i,"kind4qty13"],s6.loc[i,"kind4qty14"]]),np.max([s6.loc[i,"kind4qty8"],s6.loc[i,"kind4qty9"],s6.loc[i,"kind4qty10"],s6.loc[i,"kind4qty11"],s6.loc[i,"kind4qty12"],s6.loc[i,"kind4qty13"],s6.loc[i,"kind4qty14"]]),np.min([s6.loc[i,"kind4qty8"],s6.loc[i,"kind4qty9"],s6.loc[i,"kind4qty10"],s6.loc[i,"kind4qty11"],s6.loc[i,"kind4qty12"],s6.loc[i,"kind4qty13"],s6.loc[i,"kind4qty14"]]),np.mean([s6.loc[i,"kind4qty15"],s6.loc[i,"kind4qty16"],s6.loc[i,"kind4qty17"],s6.loc[i,"kind4qty18"],s6.loc[i,"kind4qty19"],s6.loc[i,"kind4qty20"],s6.loc[i,"kind4qty21"]]),np.max([s6.loc[i,"kind4qty15"],s6.loc[i,"kind4qty16"],s6.loc[i,"kind4qty17"],s6.loc[i,"kind4qty18"],s6.loc[i,"kind4qty19"],s6.loc[i,"kind4qty20"],s6.loc[i,"kind4qty21"]]),np.min([s6.loc[i,"kind4qty15"],s6.loc[i,"kind4qty16"],s6.loc[i,"kind4qty17"],s6.loc[i,"kind4qty18"],s6.loc[i,"kind4qty19"],s6.loc[i,"kind4qty20"],s6.loc[i,"kind4qty21"]]),np.mean([s6.loc[i,"kind4qty22"],s6.loc[i,"kind4qty23"],s6.loc[i,"kind4qty24"],s6.loc[i,"kind4qty25"],s6.loc[i,"kind4qty26"],s6.loc[i,"kind4qty27"],s6.loc[i,"kind4qty28"]]),np.max([s6.loc[i,"kind4qty22"],s6.loc[i,"kind4qty23"],s6.loc[i,"kind4qty24"],s6.loc[i,"kind4qty25"],s6.loc[i,"kind4qty26"],s6.loc[i,"kind4qty27"],s6.loc[i,"kind4qty28"]]),np.min([s6.loc[i,"kind4qty22"],s6.loc[i,"kind4qty23"],s6.loc[i,"kind4qty24"],s6.loc[i,"kind4qty25"],s6.loc[i,"kind4qty26"],s6.loc[i,"kind4qty27"],s6.loc[i,"kind4qty28"]])]
    print(i)
feature6


# In[470]:


feature6.to_csv("feature6.csv")


# In[353]:


# for i in range(df_plu.shape[1]):
#     print(i)
#     pluno = plu_list[i]
#     bndno = bnd_dict.get(plu_list[i])['bndno']
#     kind1 = (plu_list[i]/1000000).astype(int)
#     kind2 = (plu_list[i]/100000).astype(int)
#     kind3 = (plu_list[i]/10000).astype(int)
#     kind4 = (plu_list[i]/1000).astype(int)
#     sldatime = sldatime_list[j]
#     for j in range(df_plu.shape[0]):
# #         print(j)
#         index_num = i*sldatime_list.__len__()+j
#         ndf.loc[index_num,"pluno"]=pluno
#         ndf.loc[index_num,"bndno"]=bndno
#         ndf.loc[index_num,"kind1"]=kind1
#         ndf.loc[index_num,"kind2"]=kind2
#         ndf.loc[index_num,"kind3"]=kind3
#         ndf.loc[index_num,"kind4"]=kind4
#         ndf.loc[index_num,"sldatime"]=sldatime
#         ndf.loc[index_num,"isWeekday"]=not (j%7==5 or j%7==6)
#         ndf.loc[index_num,"qty"]=df_plu.iloc[j,i]
#         if j == 0:
#             continue
# #             ndf.loc[index_num,"qty1"]=0
# #             ndf.loc[index_num,"qty2"]=0
# #             ndf.loc[index_num,"qty3"]=0
# #             ndf.loc[index_num,"qty4"]=0
# #             ndf.loc[index_num,"qty5"]=0
# #             ndf.loc[index_num,"qty6"]=0
# #             ndf.loc[index_num,"qty7"]=0
# #             ndf.loc[index_num,"bndqty1"]=0
# #             ndf.loc[index_num,"bndqty2"]=0
# #             ndf.loc[index_num,"bndqty3"]=0
# #             ndf.loc[index_num,"bndqty4"]=0
# #             ndf.loc[index_num,"bndqty5"]=0
# #             ndf.loc[index_num,"bndqty6"]=0
# #             ndf.loc[index_num,"bndqty7"]=0
# #             ndf.loc[index_num,"kind1qty1"]=0
# #             ndf.loc[index_num,"kind1qty2"]=0
# #             ndf.loc[index_num,"kind1qty3"]=0
# #             ndf.loc[index_num,"kind1qty4"]=0
# #             ndf.loc[index_num,"kind1qty5"]=0
# #             ndf.loc[index_num,"kind1qty6"]=0
# #             ndf.loc[index_num,"kind1qty7"]=0
# #             ndf.loc[index_num,"kind2qty1"]=0
# #             ndf.loc[index_num,"kind2qty2"]=0
# #             ndf.loc[index_num,"kind2qty3"]=0
# #             ndf.loc[index_num,"kind2qty4"]=0
# #             ndf.loc[index_num,"kind2qty5"]=0
# #             ndf.loc[index_num,"kind2qty6"]=0
# #             ndf.loc[index_num,"kind2qty7"]=0
# #             ndf.loc[index_num,"kind3qty1"]=0
# #             ndf.loc[index_num,"kind3qty2"]=0
# #             ndf.loc[index_num,"kind3qty3"]=0
# #             ndf.loc[index_num,"kind3qty4"]=0
# #             ndf.loc[index_num,"kind3qty5"]=0
# #             ndf.loc[index_num,"kind3qty6"]=0
# #             ndf.loc[index_num,"kind3qty7"]=0
# #             ndf.loc[index_num,"kind4qty1"]=0
# #             ndf.loc[index_num,"kind4qty2"]=0
# #             ndf.loc[index_num,"kind4qty3"]=0
# #             ndf.loc[index_num,"kind4qty4"]=0
# #             ndf.loc[index_num,"kind4qty5"]=0
# #             ndf.loc[index_num,"kind4qty6"]=0
# #             ndf.loc[index_num,"kind4qty7"]=0
# #             ndf.loc[index_num,"week2AvgQty"]=0
# #             ndf.loc[index_num,"week2MaxQty"]=0
# #             ndf.loc[index_num,"week2MinQty"]=0
# #             ndf.loc[index_num,"week3AvgQty"]=0
# #             ndf.loc[index_num,"week3MaxQty"]=0
# #             ndf.loc[index_num,"week3MinQty"]=0
# #             ndf.loc[index_num,"week4AvgQty"]=0
# #             ndf.loc[index_num,"week4MaxQty"]=0
# #             ndf.loc[index_num,"week4MinQty"]=0
# #             ndf.loc[index_num,"week2AvgBndQty"]=0
# #             ndf.loc[index_num,"week2MaxBndQty"]=0
# #             ndf.loc[index_num,"week2MinBndQty"]=0
# #             ndf.loc[index_num,"week3AvgBndQty"]=0
# #             ndf.loc[index_num,"week3MaxBndQty"]=0
# #             ndf.loc[index_num,"week3MinBndQty"]=0
# #             ndf.loc[index_num,"week4AvgBndQty"]=0
# #             ndf.loc[index_num,"week4MaxBndQty"]=0
# #             ndf.loc[index_num,"week4MinBndQty"]=0
# #             ndf.loc[index_num,"week2AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week2MinKind1Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week3MinKind1Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week4MinKind1Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week2MinKind2Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week3MinKind2Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week4MinKind2Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week2MinKind3Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week3MinKind3Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week4MinKind3Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week2MinKind4Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week3MinKind4Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         elif j == 1:
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
# #             ndf.loc[index_num,"qty2"]=0
# #             ndf.loc[index_num,"qty3"]=0
# #             ndf.loc[index_num,"qty4"]=0
# #             ndf.loc[index_num,"qty5"]=0
# #             ndf.loc[index_num,"qty6"]=0
# #             ndf.loc[index_num,"qty7"]=0
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
# #             ndf.loc[index_num,"bndqty2"]=0
# #             ndf.loc[index_num,"bndqty3"]=0
# #             ndf.loc[index_num,"bndqty4"]=0
# #             ndf.loc[index_num,"bndqty5"]=0
# #             ndf.loc[index_num,"bndqty6"]=0
# #             ndf.loc[index_num,"bndqty7"]=0
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
# #             ndf.loc[index_num,"kind1qty2"]=0
# #             ndf.loc[index_num,"kind1qty3"]=0
# #             ndf.loc[index_num,"kind1qty4"]=0
# #             ndf.loc[index_num,"kind1qty5"]=0
# #             ndf.loc[index_num,"kind1qty6"]=0
# #             ndf.loc[index_num,"kind1qty7"]=0
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
# #             ndf.loc[index_num,"kind2qty2"]=0
# #             ndf.loc[index_num,"kind2qty3"]=0
# #             ndf.loc[index_num,"kind2qty4"]=0
# #             ndf.loc[index_num,"kind2qty5"]=0
# #             ndf.loc[index_num,"kind2qty6"]=0
# #             ndf.loc[index_num,"kind2qty7"]=0
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime_list[j-1],str(kind3)]
# #             ndf.loc[index_num,"kind3qty2"]=0
# #             ndf.loc[index_num,"kind3qty3"]=0
# #             ndf.loc[index_num,"kind3qty4"]=0
# #             ndf.loc[index_num,"kind3qty5"]=0
# #             ndf.loc[index_num,"kind3qty6"]=0
# #             ndf.loc[index_num,"kind3qty7"]=0
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime_list[j-1],str(kind4)]
# #             ndf.loc[index_num,"kind4qty2"]=0
# #             ndf.loc[index_num,"kind4qty3"]=0
# #             ndf.loc[index_num,"kind4qty4"]=0
# #             ndf.loc[index_num,"kind4qty5"]=0
# #             ndf.loc[index_num,"kind4qty6"]=0
# #             ndf.loc[index_num,"kind4qty7"]=0
# #             ndf.loc[index_num,"week2AvgQty"]=0
# #             ndf.loc[index_num,"week2MaxQty"]=0
# #             ndf.loc[index_num,"week2MinQty"]=0
# #             ndf.loc[index_num,"week3AvgQty"]=0
# #             ndf.loc[index_num,"week3MaxQty"]=0
# #             ndf.loc[index_num,"week3MinQty"]=0
# #             ndf.loc[index_num,"week4AvgQty"]=0
# #             ndf.loc[index_num,"week4MaxQty"]=0
# #             ndf.loc[index_num,"week4MinQty"]=0
# #             ndf.loc[index_num,"week2AvgBndQty"]=0
# #             ndf.loc[index_num,"week2MaxBndQty"]=0
# #             ndf.loc[index_num,"week2MinBndQty"]=0
# #             ndf.loc[index_num,"week3AvgBndQty"]=0
# #             ndf.loc[index_num,"week3MaxBndQty"]=0
# #             ndf.loc[index_num,"week3MinBndQty"]=0
# #             ndf.loc[index_num,"week4AvgBndQty"]=0
# #             ndf.loc[index_num,"week4MaxBndQty"]=0
# #             ndf.loc[index_num,"week4MinBndQty"]=0
# #             ndf.loc[index_num,"week2AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week2MinKind1Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week3MinKind1Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week4MinKind1Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week2MinKind2Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week3MinKind2Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week4MinKind2Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week2MinKind3Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week3MinKind3Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week4MinKind3Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week2MinKind4Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week3MinKind4Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         elif j == 2:
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
#             ndf.loc[index_num,"qty2"]=df_plu.iloc[j-2,i]
# #             ndf.loc[index_num,"qty3"]=0
# #             ndf.loc[index_num,"qty4"]=0
# #             ndf.loc[index_num,"qty5"]=0
# #             ndf.loc[index_num,"qty6"]=0
# #             ndf.loc[index_num,"qty7"]=0
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
#             ndf.loc[index_num,"bndqty2"]=df_bnd.loc[sldatime_list[j-2],str(bndno)]
# #             ndf.loc[index_num,"bndqty3"]=0
# #             ndf.loc[index_num,"bndqty4"]=0
# #             ndf.loc[index_num,"bndqty5"]=0
# #             ndf.loc[index_num,"bndqty6"]=0
# #             ndf.loc[index_num,"bndqty7"]=0
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
#             ndf.loc[index_num,"kind1qty2"]=df_kind1.loc[sldatime_list[j-2],str(kind1)]
# #             ndf.loc[index_num,"kind1qty3"]=0
# #             ndf.loc[index_num,"kind1qty4"]=0
# #             ndf.loc[index_num,"kind1qty5"]=0
# #             ndf.loc[index_num,"kind1qty6"]=0
# #             ndf.loc[index_num,"kind1qty7"]=0
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
#             ndf.loc[index_num,"kind2qty2"]=df_kind2.loc[sldatime_list[j-2],str(kind2)]
# #             ndf.loc[index_num,"kind2qty3"]=0
# #             ndf.loc[index_num,"kind2qty4"]=0
# #             ndf.loc[index_num,"kind2qty5"]=0
# #             ndf.loc[index_num,"kind2qty6"]=0
# #             ndf.loc[index_num,"kind2qty7"]=0
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime_list[j-1],str(kind3)]
#             ndf.loc[index_num,"kind3qty2"]=df_kind3.loc[sldatime_list[j-2],str(kind3)]
# #             ndf.loc[index_num,"kind3qty3"]=0
# #             ndf.loc[index_num,"kind3qty4"]=0
# #             ndf.loc[index_num,"kind3qty5"]=0
# #             ndf.loc[index_num,"kind3qty6"]=0
# #             ndf.loc[index_num,"kind3qty7"]=0
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime_list[j-1],str(kind4)]
#             ndf.loc[index_num,"kind4qty2"]=df_kind4.loc[sldatime_list[j-2],str(kind4)]
# #             ndf.loc[index_num,"kind4qty3"]=0
# #             ndf.loc[index_num,"kind4qty4"]=0
# #             ndf.loc[index_num,"kind4qty5"]=0
# #             ndf.loc[index_num,"kind4qty6"]=0
# #             ndf.loc[index_num,"kind4qty7"]=0
# #             ndf.loc[index_num,"week2AvgQty"]=0
# #             ndf.loc[index_num,"week2MaxQty"]=0
# #             ndf.loc[index_num,"week2MinQty"]=0
# #             ndf.loc[index_num,"week3AvgQty"]=0
# #             ndf.loc[index_num,"week3MaxQty"]=0
# #             ndf.loc[index_num,"week3MinQty"]=0
# #             ndf.loc[index_num,"week4AvgQty"]=0
# #             ndf.loc[index_num,"week4MaxQty"]=0
# #             ndf.loc[index_num,"week4MinQty"]=0
# #             ndf.loc[index_num,"week2AvgBndQty"]=0
# #             ndf.loc[index_num,"week2MaxBndQty"]=0
# #             ndf.loc[index_num,"week2MinBndQty"]=0
# #             ndf.loc[index_num,"week3AvgBndQty"]=0
# #             ndf.loc[index_num,"week3MaxBndQty"]=0
# #             ndf.loc[index_num,"week3MinBndQty"]=0
# #             ndf.loc[index_num,"week4AvgBndQty"]=0
# #             ndf.loc[index_num,"week4MaxBndQty"]=0
# #             ndf.loc[index_num,"week4MinBndQty"]=0
# #             ndf.loc[index_num,"week2AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week2MinKind1Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week3MinKind1Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week4MinKind1Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week2MinKind2Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week3MinKind2Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week4MinKind2Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week2MinKind3Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week3MinKind3Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week4MinKind3Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week2MinKind4Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week3MinKind4Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         elif j == 3:
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
#             ndf.loc[index_num,"qty2"]=df_plu.iloc[j-2,i]
#             ndf.loc[index_num,"qty3"]=df_plu.iloc[j-3,i]
# #             ndf.loc[index_num,"qty4"]=0
# #             ndf.loc[index_num,"qty5"]=0
# #             ndf.loc[index_num,"qty6"]=0
# #             ndf.loc[index_num,"qty7"]=0
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
#             ndf.loc[index_num,"bndqty2"]=df_bnd.loc[sldatime_list[j-2],str(bndno)]
#             ndf.loc[index_num,"bndqty3"]=df_bnd.loc[sldatime_list[j-3],str(bndno)]
# #             ndf.loc[index_num,"bndqty4"]=0
# #             ndf.loc[index_num,"bndqty5"]=0
# #             ndf.loc[index_num,"bndqty6"]=0
# #             ndf.loc[index_num,"bndqty7"]=0
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
#             ndf.loc[index_num,"kind1qty2"]=df_kind1.loc[sldatime_list[j-2],str(kind1)]
#             ndf.loc[index_num,"kind1qty3"]=df_kind1.loc[sldatime_list[j-3],str(kind1)]
# #             ndf.loc[index_num,"kind1qty4"]=0
# #             ndf.loc[index_num,"kind1qty5"]=0
# #             ndf.loc[index_num,"kind1qty6"]=0
# #             ndf.loc[index_num,"kind1qty7"]=0
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
#             ndf.loc[index_num,"kind2qty2"]=df_kind2.loc[sldatime_list[j-2],str(kind2)]
#             ndf.loc[index_num,"kind2qty3"]=df_kind2.loc[sldatime_list[j-3],str(kind2)]
# #             ndf.loc[index_num,"kind2qty4"]=0
# #             ndf.loc[index_num,"kind2qty5"]=0
# #             ndf.loc[index_num,"kind2qty6"]=0
# #             ndf.loc[index_num,"kind2qty7"]=0
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime_list[j-1],str(kind3)]
#             ndf.loc[index_num,"kind3qty2"]=df_kind3.loc[sldatime_list[j-2],str(kind3)]
#             ndf.loc[index_num,"kind3qty3"]=df_kind3.loc[sldatime_list[j-3],str(kind3)]
# #             ndf.loc[index_num,"kind3qty4"]=0
# #             ndf.loc[index_num,"kind3qty5"]=0
# #             ndf.loc[index_num,"kind3qty6"]=0
# #             ndf.loc[index_num,"kind3qty7"]=0
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime_list[j-1],str(kind4)]
#             ndf.loc[index_num,"kind4qty2"]=df_kind4.loc[sldatime_list[j-2],str(kind4)]
#             ndf.loc[index_num,"kind4qty3"]=df_kind4.loc[sldatime_list[j-3],str(kind4)]
# #             ndf.loc[index_num,"kind4qty4"]=0
# #             ndf.loc[index_num,"kind4qty5"]=0
# #             ndf.loc[index_num,"kind4qty6"]=0
# #             ndf.loc[index_num,"kind4qty7"]=0
# #             ndf.loc[index_num,"week2AvgQty"]=0
# #             ndf.loc[index_num,"week2MaxQty"]=0
# #             ndf.loc[index_num,"week2MinQty"]=0
# #             ndf.loc[index_num,"week3AvgQty"]=0
# #             ndf.loc[index_num,"week3MaxQty"]=0
# #             ndf.loc[index_num,"week3MinQty"]=0
# #             ndf.loc[index_num,"week4AvgQty"]=0
# #             ndf.loc[index_num,"week4MaxQty"]=0
# #             ndf.loc[index_num,"week4MinQty"]=0
# #             ndf.loc[index_num,"week2AvgBndQty"]=0
# #             ndf.loc[index_num,"week2MaxBndQty"]=0
# #             ndf.loc[index_num,"week2MinBndQty"]=0
# #             ndf.loc[index_num,"week3AvgBndQty"]=0
# #             ndf.loc[index_num,"week3MaxBndQty"]=0
# #             ndf.loc[index_num,"week3MinBndQty"]=0
# #             ndf.loc[index_num,"week4AvgBndQty"]=0
# #             ndf.loc[index_num,"week4MaxBndQty"]=0
# #             ndf.loc[index_num,"week4MinBndQty"]=0
# #             ndf.loc[index_num,"week2AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week2MinKind1Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week3MinKind1Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week4MinKind1Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week2MinKind2Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week3MinKind2Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week4MinKind2Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week2MinKind3Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week3MinKind3Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week4MinKind3Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week2MinKind4Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week3MinKind4Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         elif j == 4:
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
#             ndf.loc[index_num,"qty2"]=df_plu.iloc[j-2,i]
#             ndf.loc[index_num,"qty3"]=df_plu.iloc[j-3,i]
#             ndf.loc[index_num,"qty4"]=df_plu.iloc[j-4,i]
# #             ndf.loc[index_num,"qty5"]=0
# #             ndf.loc[index_num,"qty6"]=0
# #             ndf.loc[index_num,"qty7"]=0
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
#             ndf.loc[index_num,"bndqty2"]=df_bnd.loc[sldatime_list[j-2],str(bndno)]
#             ndf.loc[index_num,"bndqty3"]=df_bnd.loc[sldatime_list[j-3],str(bndno)]
#             ndf.loc[index_num,"bndqty4"]=df_bnd.loc[sldatime_list[j-4],str(bndno)]
# #             ndf.loc[index_num,"bndqty5"]=0
# #             ndf.loc[index_num,"bndqty6"]=0
# #             ndf.loc[index_num,"bndqty7"]=0
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
#             ndf.loc[index_num,"kind1qty2"]=df_kind1.loc[sldatime_list[j-2],str(kind1)]
#             ndf.loc[index_num,"kind1qty3"]=df_kind1.loc[sldatime_list[j-3],str(kind1)]
#             ndf.loc[index_num,"kind1qty4"]=df_kind1.loc[sldatime_list[j-4],str(kind1)]
# #             ndf.loc[index_num,"kind1qty5"]=0
# #             ndf.loc[index_num,"kind1qty6"]=0
# #             ndf.loc[index_num,"kind1qty7"]=0
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
#             ndf.loc[index_num,"kind2qty2"]=df_kind2.loc[sldatime_list[j-2],str(kind2)]
#             ndf.loc[index_num,"kind2qty3"]=df_kind2.loc[sldatime_list[j-3],str(kind2)]
#             ndf.loc[index_num,"kind2qty4"]=df_kind2.loc[sldatime_list[j-4],str(kind2)]
# #             ndf.loc[index_num,"kind2qty5"]=0
# #             ndf.loc[index_num,"kind2qty6"]=0
# #             ndf.loc[index_num,"kind2qty7"]=0
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime_list[j-1],str(kind3)]
#             ndf.loc[index_num,"kind3qty2"]=df_kind3.loc[sldatime_list[j-2],str(kind3)]
#             ndf.loc[index_num,"kind3qty3"]=df_kind3.loc[sldatime_list[j-3],str(kind3)]
#             ndf.loc[index_num,"kind3qty4"]=df_kind3.loc[sldatime_list[j-4],str(kind3)]
# #             ndf.loc[index_num,"kind3qty5"]=0
# #             ndf.loc[index_num,"kind3qty6"]=0
# #             ndf.loc[index_num,"kind3qty7"]=0
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime_list[j-1],str(kind4)]
#             ndf.loc[index_num,"kind4qty2"]=df_kind4.loc[sldatime_list[j-2],str(kind4)]
#             ndf.loc[index_num,"kind4qty3"]=df_kind4.loc[sldatime_list[j-3],str(kind4)]
#             ndf.loc[index_num,"kind4qty4"]=df_kind4.loc[sldatime_list[j-4],str(kind4)]
# #             ndf.loc[index_num,"kind4qty5"]=0
# #             ndf.loc[index_num,"kind4qty6"]=0
# #             ndf.loc[index_num,"kind4qty7"]=0
# #             ndf.loc[index_num,"week2AvgQty"]=0
# #             ndf.loc[index_num,"week2MaxQty"]=0
# #             ndf.loc[index_num,"week2MinQty"]=0
# #             ndf.loc[index_num,"week3AvgQty"]=0
# #             ndf.loc[index_num,"week3MaxQty"]=0
# #             ndf.loc[index_num,"week3MinQty"]=0
# #             ndf.loc[index_num,"week4AvgQty"]=0
# #             ndf.loc[index_num,"week4MaxQty"]=0
# #             ndf.loc[index_num,"week4MinQty"]=0
# #             ndf.loc[index_num,"week2AvgBndQty"]=0
# #             ndf.loc[index_num,"week2MaxBndQty"]=0
# #             ndf.loc[index_num,"week2MinBndQty"]=0
# #             ndf.loc[index_num,"week3AvgBndQty"]=0
# #             ndf.loc[index_num,"week3MaxBndQty"]=0
# #             ndf.loc[index_num,"week3MinBndQty"]=0
# #             ndf.loc[index_num,"week4AvgBndQty"]=0
# #             ndf.loc[index_num,"week4MaxBndQty"]=0
# #             ndf.loc[index_num,"week4MinBndQty"]=0
# #             ndf.loc[index_num,"week2AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week2MinKind1Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week3MinKind1Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind1Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind1Qty"]=0
# #             ndf.loc[index_num,"week4MinKind1Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week2MinKind2Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week3MinKind2Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind2Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind2Qty"]=0
# #             ndf.loc[index_num,"week4MinKind2Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week2MinKind3Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week3MinKind3Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind3Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind3Qty"]=0
# #             ndf.loc[index_num,"week4MinKind3Qty"]=0
# #             ndf.loc[index_num,"week2AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week2MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week2MinKind4Qty"]=0
# #             ndf.loc[index_num,"week3AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week3MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week3MinKind4Qty"]=0
# #             ndf.loc[index_num,"week4AvgKind4Qty"]=0
# #             ndf.loc[index_num,"week4MaxKind4Qty"]=0
# #             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         elif j == 5:
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
#             ndf.loc[index_num,"qty2"]=df_plu.iloc[j-2,i]
#             ndf.loc[index_num,"qty3"]=df_plu.iloc[j-3,i]
#             ndf.loc[index_num,"qty4"]=df_plu.iloc[j-4,i]
#             ndf.loc[index_num,"qty5"]=df_plu.iloc[j-5,i]
# #             ndf.loc[index_num,"qty6"]=0
# #             ndf.loc[index_num,"qty7"]=0
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
#             ndf.loc[index_num,"bndqty2"]=df_bnd.loc[sldatime_list[j-2],str(bndno)]
#             ndf.loc[index_num,"bndqty3"]=df_bnd.loc[sldatime_list[j-3],str(bndno)]
#             ndf.loc[index_num,"bndqty4"]=df_bnd.loc[sldatime_list[j-4],str(bndno)]
#             ndf.loc[index_num,"bndqty5"]=df_bnd.loc[sldatime_list[j-5],str(bndno)]
# #             ndf.loc[index_num,"bndqty6"]=0
# #             ndf.loc[index_num,"bndqty7"]=0
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
#             ndf.loc[index_num,"kind1qty2"]=df_kind1.loc[sldatime_list[j-2],str(kind1)]
#             ndf.loc[index_num,"kind1qty3"]=df_kind1.loc[sldatime_list[j-3],str(kind1)]
#             ndf.loc[index_num,"kind1qty4"]=df_kind1.loc[sldatime_list[j-4],str(kind1)]
#             ndf.loc[index_num,"kind1qty5"]=df_kind1.loc[sldatime_list[j-5],str(kind1)]
# #             ndf.loc[index_num,"kind1qty6"]=0
# #             ndf.loc[index_num,"kind1qty7"]=0
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
#             ndf.loc[index_num,"kind2qty2"]=df_kind2.loc[sldatime_list[j-2],str(kind2)]
#             ndf.loc[index_num,"kind2qty3"]=df_kind2.loc[sldatime_list[j-3],str(kind2)]
#             ndf.loc[index_num,"kind2qty4"]=df_kind2.loc[sldatime_list[j-4],str(kind2)]
#             ndf.loc[index_num,"kind2qty5"]=df_kind2.loc[sldatime_list[j-5],str(kind2)]
# #             ndf.loc[index_num,"kind2qty6"]=0
# #             ndf.loc[index_num,"kind2qty7"]=0
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime_list[j-1],str(kind3)]
#             ndf.loc[index_num,"kind3qty2"]=df_kind3.loc[sldatime_list[j-2],str(kind3)]
#             ndf.loc[index_num,"kind3qty3"]=df_kind3.loc[sldatime_list[j-3],str(kind3)]
#             ndf.loc[index_num,"kind3qty4"]=df_kind3.loc[sldatime_list[j-4],str(kind3)]
#             ndf.loc[index_num,"kind3qty5"]=df_kind3.loc[sldatime_list[j-5],str(kind3)]
# #             ndf.loc[index_num,"kind3qty6"]=0
# #             ndf.loc[index_num,"kind3qty7"]=0
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime_list[j-1],str(kind4)]
#             ndf.loc[index_num,"kind4qty2"]=df_kind4.loc[sldatime_list[j-2],str(kind4)]
#             ndf.loc[index_num,"kind4qty3"]=df_kind4.loc[sldatime_list[j-3],str(kind4)]
#             ndf.loc[index_num,"kind4qty4"]=df_kind4.loc[sldatime_list[j-4],str(kind4)]
#             ndf.loc[index_num,"kind4qty5"]=df_kind4.loc[sldatime_list[j-5],str(kind4)]
#             ndf.loc[index_num,"kind4qty6"]=0
#             ndf.loc[index_num,"kind4qty7"]=0
#             ndf.loc[index_num,"week2AvgQty"]=0
#             ndf.loc[index_num,"week2MaxQty"]=0
#             ndf.loc[index_num,"week2MinQty"]=0
#             ndf.loc[index_num,"week3AvgQty"]=0
#             ndf.loc[index_num,"week3MaxQty"]=0
#             ndf.loc[index_num,"week3MinQty"]=0
#             ndf.loc[index_num,"week4AvgQty"]=0
#             ndf.loc[index_num,"week4MaxQty"]=0
#             ndf.loc[index_num,"week4MinQty"]=0
#             ndf.loc[index_num,"week2AvgBndQty"]=0
#             ndf.loc[index_num,"week2MaxBndQty"]=0
#             ndf.loc[index_num,"week2MinBndQty"]=0
#             ndf.loc[index_num,"week3AvgBndQty"]=0
#             ndf.loc[index_num,"week3MaxBndQty"]=0
#             ndf.loc[index_num,"week3MinBndQty"]=0
#             ndf.loc[index_num,"week4AvgBndQty"]=0
#             ndf.loc[index_num,"week4MaxBndQty"]=0
#             ndf.loc[index_num,"week4MinBndQty"]=0
#             ndf.loc[index_num,"week2AvgKind1Qty"]=0
#             ndf.loc[index_num,"week2MaxKind1Qty"]=0
#             ndf.loc[index_num,"week2MinKind1Qty"]=0
#             ndf.loc[index_num,"week3AvgKind1Qty"]=0
#             ndf.loc[index_num,"week3MaxKind1Qty"]=0
#             ndf.loc[index_num,"week3MinKind1Qty"]=0
#             ndf.loc[index_num,"week4AvgKind1Qty"]=0
#             ndf.loc[index_num,"week4MaxKind1Qty"]=0
#             ndf.loc[index_num,"week4MinKind1Qty"]=0
#             ndf.loc[index_num,"week2AvgKind2Qty"]=0
#             ndf.loc[index_num,"week2MaxKind2Qty"]=0
#             ndf.loc[index_num,"week2MinKind2Qty"]=0
#             ndf.loc[index_num,"week3AvgKind2Qty"]=0
#             ndf.loc[index_num,"week3MaxKind2Qty"]=0
#             ndf.loc[index_num,"week3MinKind2Qty"]=0
#             ndf.loc[index_num,"week4AvgKind2Qty"]=0
#             ndf.loc[index_num,"week4MaxKind2Qty"]=0
#             ndf.loc[index_num,"week4MinKind2Qty"]=0
#             ndf.loc[index_num,"week2AvgKind3Qty"]=0
#             ndf.loc[index_num,"week2MaxKind3Qty"]=0
#             ndf.loc[index_num,"week2MinKind3Qty"]=0
#             ndf.loc[index_num,"week3AvgKind3Qty"]=0
#             ndf.loc[index_num,"week3MaxKind3Qty"]=0
#             ndf.loc[index_num,"week3MinKind3Qty"]=0
#             ndf.loc[index_num,"week4AvgKind3Qty"]=0
#             ndf.loc[index_num,"week4MaxKind3Qty"]=0
#             ndf.loc[index_num,"week4MinKind3Qty"]=0
#             ndf.loc[index_num,"week2AvgKind4Qty"]=0
#             ndf.loc[index_num,"week2MaxKind4Qty"]=0
#             ndf.loc[index_num,"week2MinKind4Qty"]=0
#             ndf.loc[index_num,"week3AvgKind4Qty"]=0
#             ndf.loc[index_num,"week3MaxKind4Qty"]=0
#             ndf.loc[index_num,"week3MinKind4Qty"]=0
#             ndf.loc[index_num,"week4AvgKind4Qty"]=0
#             ndf.loc[index_num,"week4MaxKind4Qty"]=0
#             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         elif j == 6:
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
#             ndf.loc[index_num,"qty2"]=df_plu.iloc[j-2,i]
#             ndf.loc[index_num,"qty3"]=df_plu.iloc[j-3,i]
#             ndf.loc[index_num,"qty4"]=df_plu.iloc[j-4,i]
#             ndf.loc[index_num,"qty5"]=df_plu.iloc[j-5,i]
#             ndf.loc[index_num,"qty6"]=df_plu.iloc[j-6,i]
#             ndf.loc[index_num,"qty7"]=0
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
#             ndf.loc[index_num,"bndqty2"]=df_bnd.loc[sldatime_list[j-2],str(bndno)]
#             ndf.loc[index_num,"bndqty3"]=df_bnd.loc[sldatime_list[j-3],str(bndno)]
#             ndf.loc[index_num,"bndqty4"]=df_bnd.loc[sldatime_list[j-4],str(bndno)]
#             ndf.loc[index_num,"bndqty5"]=df_bnd.loc[sldatime_list[j-5],str(bndno)]
#             ndf.loc[index_num,"bndqty6"]=df_bnd.loc[sldatime_list[j-6],str(bndno)]
#             ndf.loc[index_num,"bndqty7"]=0
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
#             ndf.loc[index_num,"kind1qty2"]=df_kind1.loc[sldatime_list[j-2],str(kind1)]
#             ndf.loc[index_num,"kind1qty3"]=df_kind1.loc[sldatime_list[j-3],str(kind1)]
#             ndf.loc[index_num,"kind1qty4"]=df_kind1.loc[sldatime_list[j-4],str(kind1)]
#             ndf.loc[index_num,"kind1qty5"]=df_kind1.loc[sldatime_list[j-5],str(kind1)]
#             ndf.loc[index_num,"kind1qty6"]=df_kind1.loc[sldatime_list[j-6],str(kind1)]
#             ndf.loc[index_num,"kind1qty7"]=0
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
#             ndf.loc[index_num,"kind2qty2"]=df_kind2.loc[sldatime_list[j-2],str(kind2)]
#             ndf.loc[index_num,"kind2qty3"]=df_kind2.loc[sldatime_list[j-3],str(kind2)]
#             ndf.loc[index_num,"kind2qty4"]=df_kind2.loc[sldatime_list[j-4],str(kind2)]
#             ndf.loc[index_num,"kind2qty5"]=df_kind2.loc[sldatime_list[j-5],str(kind2)]
#             ndf.loc[index_num,"kind2qty6"]=df_kind2.loc[sldatime_list[j-6],str(kind2)]
#             ndf.loc[index_num,"kind2qty7"]=0
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime_list[j-1],str(kind3)]
#             ndf.loc[index_num,"kind3qty2"]=df_kind3.loc[sldatime_list[j-2],str(kind3)]
#             ndf.loc[index_num,"kind3qty3"]=df_kind3.loc[sldatime_list[j-3],str(kind3)]
#             ndf.loc[index_num,"kind3qty4"]=df_kind3.loc[sldatime_list[j-4],str(kind3)]
#             ndf.loc[index_num,"kind3qty5"]=df_kind3.loc[sldatime_list[j-5],str(kind3)]
#             ndf.loc[index_num,"kind3qty6"]=df_kind3.loc[sldatime_list[j-6],str(kind3)]
#             ndf.loc[index_num,"kind3qty7"]=0
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime_list[j-1],str(kind4)]
#             ndf.loc[index_num,"kind4qty2"]=df_kind4.loc[sldatime_list[j-2],str(kind4)]
#             ndf.loc[index_num,"kind4qty3"]=df_kind4.loc[sldatime_list[j-3],str(kind4)]
#             ndf.loc[index_num,"kind4qty4"]=df_kind4.loc[sldatime_list[j-4],str(kind4)]
#             ndf.loc[index_num,"kind4qty5"]=df_kind4.loc[sldatime_list[j-5],str(kind4)]
#             ndf.loc[index_num,"kind4qty6"]=df_kind4.loc[sldatime_list[j-6],str(kind4)]
#             ndf.loc[index_num,"kind4qty7"]=0
#             ndf.loc[index_num,"week2AvgQty"]=0
#             ndf.loc[index_num,"week2MaxQty"]=0
#             ndf.loc[index_num,"week2MinQty"]=0
#             ndf.loc[index_num,"week3AvgQty"]=0
#             ndf.loc[index_num,"week3MaxQty"]=0
#             ndf.loc[index_num,"week3MinQty"]=0
#             ndf.loc[index_num,"week4AvgQty"]=0
#             ndf.loc[index_num,"week4MaxQty"]=0
#             ndf.loc[index_num,"week4MinQty"]=0
#             ndf.loc[index_num,"week2AvgBndQty"]=0
#             ndf.loc[index_num,"week2MaxBndQty"]=0
#             ndf.loc[index_num,"week2MinBndQty"]=0
#             ndf.loc[index_num,"week3AvgBndQty"]=0
#             ndf.loc[index_num,"week3MaxBndQty"]=0
#             ndf.loc[index_num,"week3MinBndQty"]=0
#             ndf.loc[index_num,"week4AvgBndQty"]=0
#             ndf.loc[index_num,"week4MaxBndQty"]=0
#             ndf.loc[index_num,"week4MinBndQty"]=0
#             ndf.loc[index_num,"week2AvgKind1Qty"]=0
#             ndf.loc[index_num,"week2MaxKind1Qty"]=0
#             ndf.loc[index_num,"week2MinKind1Qty"]=0
#             ndf.loc[index_num,"week3AvgKind1Qty"]=0
#             ndf.loc[index_num,"week3MaxKind1Qty"]=0
#             ndf.loc[index_num,"week3MinKind1Qty"]=0
#             ndf.loc[index_num,"week4AvgKind1Qty"]=0
#             ndf.loc[index_num,"week4MaxKind1Qty"]=0
#             ndf.loc[index_num,"week4MinKind1Qty"]=0
#             ndf.loc[index_num,"week2AvgKind2Qty"]=0
#             ndf.loc[index_num,"week2MaxKind2Qty"]=0
#             ndf.loc[index_num,"week2MinKind2Qty"]=0
#             ndf.loc[index_num,"week3AvgKind2Qty"]=0
#             ndf.loc[index_num,"week3MaxKind2Qty"]=0
#             ndf.loc[index_num,"week3MinKind2Qty"]=0
#             ndf.loc[index_num,"week4AvgKind2Qty"]=0
#             ndf.loc[index_num,"week4MaxKind2Qty"]=0
#             ndf.loc[index_num,"week4MinKind2Qty"]=0
#             ndf.loc[index_num,"week2AvgKind3Qty"]=0
#             ndf.loc[index_num,"week2MaxKind3Qty"]=0
#             ndf.loc[index_num,"week2MinKind3Qty"]=0
#             ndf.loc[index_num,"week3AvgKind3Qty"]=0
#             ndf.loc[index_num,"week3MaxKind3Qty"]=0
#             ndf.loc[index_num,"week3MinKind3Qty"]=0
#             ndf.loc[index_num,"week4AvgKind3Qty"]=0
#             ndf.loc[index_num,"week4MaxKind3Qty"]=0
#             ndf.loc[index_num,"week4MinKind3Qty"]=0
#             ndf.loc[index_num,"week2AvgKind4Qty"]=0
#             ndf.loc[index_num,"week2MaxKind4Qty"]=0
#             ndf.loc[index_num,"week2MinKind4Qty"]=0
#             ndf.loc[index_num,"week3AvgKind4Qty"]=0
#             ndf.loc[index_num,"week3MaxKind4Qty"]=0
#             ndf.loc[index_num,"week3MinKind4Qty"]=0
#             ndf.loc[index_num,"week4AvgKind4Qty"]=0
#             ndf.loc[index_num,"week4MaxKind4Qty"]=0
#             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         elif j == 7:
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
#             ndf.loc[index_num,"qty2"]=df_plu.iloc[j-2,i]
#             ndf.loc[index_num,"qty3"]=df_plu.iloc[j-3,i]
#             ndf.loc[index_num,"qty4"]=df_plu.iloc[j-4,i]
#             ndf.loc[index_num,"qty5"]=df_plu.iloc[j-5,i]
#             ndf.loc[index_num,"qty6"]=df_plu.iloc[j-6,i]
#             ndf.loc[index_num,"qty7"]=df_plu.iloc[j-7,i]
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
#             ndf.loc[index_num,"bndqty2"]=df_bnd.loc[sldatime_list[j-2],str(bndno)]
#             ndf.loc[index_num,"bndqty3"]=df_bnd.loc[sldatime_list[j-3],str(bndno)]
#             ndf.loc[index_num,"bndqty4"]=df_bnd.loc[sldatime_list[j-4],str(bndno)]
#             ndf.loc[index_num,"bndqty5"]=df_bnd.loc[sldatime_list[j-5],str(bndno)]
#             ndf.loc[index_num,"bndqty6"]=df_bnd.loc[sldatime_list[j-6],str(bndno)]
#             ndf.loc[index_num,"bndqty7"]=df_bnd.loc[sldatime_list[j-7],str(bndno)]
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
#             ndf.loc[index_num,"kind1qty2"]=df_kind1.loc[sldatime_list[j-2],str(kind1)]
#             ndf.loc[index_num,"kind1qty3"]=df_kind1.loc[sldatime_list[j-3],str(kind1)]
#             ndf.loc[index_num,"kind1qty4"]=df_kind1.loc[sldatime_list[j-4],str(kind1)]
#             ndf.loc[index_num,"kind1qty5"]=df_kind1.loc[sldatime_list[j-5],str(kind1)]
#             ndf.loc[index_num,"kind1qty6"]=df_kind1.loc[sldatime_list[j-6],str(kind1)]
#             ndf.loc[index_num,"kind1qty7"]=df_kind1.loc[sldatime_list[j-7],str(kind1)]
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
#             ndf.loc[index_num,"kind2qty2"]=df_kind2.loc[sldatime_list[j-2],str(kind2)]
#             ndf.loc[index_num,"kind2qty3"]=df_kind2.loc[sldatime[j-3],str(kind2)]
#             ndf.loc[index_num,"kind2qty4"]=df_kind2.loc[sldatime[j-4],str(kind2)]
#             ndf.loc[index_num,"kind2qty5"]=df_kind2.loc[sldatime[j-5],str(kind2)]
#             ndf.loc[index_num,"kind2qty6"]=df_kind2.loc[sldatime[j-6],str(kind2)]
#             ndf.loc[index_num,"kind2qty7"]=df_kind2.loc[sldatime[j-7],str(kind2)]
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime[j-1],str(kind3)]
#             ndf.loc[index_num,"kind3qty2"]=df_kind3.loc[sldatime[j-2],str(kind3)]
#             ndf.loc[index_num,"kind3qty3"]=df_kind3.loc[sldatime[j-3],str(kind3)]
#             ndf.loc[index_num,"kind3qty4"]=df_kind3.loc[sldatime[j-4],str(kind3)]
#             ndf.loc[index_num,"kind3qty5"]=df_kind3.loc[sldatime[j-5],str(kind3)]
#             ndf.loc[index_num,"kind3qty6"]=df_kind3.loc[sldatime[j-6],str(kind3)]
#             ndf.loc[index_num,"kind3qty7"]=df_kind3.loc[sldatime[j-7],str(kind3)]
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime[j-1],str(kind4)]
#             ndf.loc[index_num,"kind4qty2"]=df_kind4.loc[sldatime[j-2],str(kind4)]
#             ndf.loc[index_num,"kind4qty3"]=df_kind4.loc[sldatime[j-3],str(kind4)]
#             ndf.loc[index_num,"kind4qty4"]=df_kind4.loc[sldatime[j-4],str(kind4)]
#             ndf.loc[index_num,"kind4qty5"]=df_kind4.loc[sldatime[j-5],str(kind4)]
#             ndf.loc[index_num,"kind4qty6"]=df_kind4.loc[sldatime[j-6],str(kind4)]
#             ndf.loc[index_num,"kind4qty7"]=df_kind4.loc[sldatime[j-7],str(kind4)]
#             ndf.loc[index_num,"week2AvgQty"]=0
#             ndf.loc[index_num,"week2MaxQty"]=0
#             ndf.loc[index_num,"week2MinQty"]=0
#             ndf.loc[index_num,"week3AvgQty"]=0
#             ndf.loc[index_num,"week3MaxQty"]=0
#             ndf.loc[index_num,"week3MinQty"]=0
#             ndf.loc[index_num,"week4AvgQty"]=0
#             ndf.loc[index_num,"week4MaxQty"]=0
#             ndf.loc[index_num,"week4MinQty"]=0
#             ndf.loc[index_num,"week2AvgBndQty"]=0
#             ndf.loc[index_num,"week2MaxBndQty"]=0
#             ndf.loc[index_num,"week2MinBndQty"]=0
#             ndf.loc[index_num,"week3AvgBndQty"]=0
#             ndf.loc[index_num,"week3MaxBndQty"]=0
#             ndf.loc[index_num,"week3MinBndQty"]=0
#             ndf.loc[index_num,"week4AvgBndQty"]=0
#             ndf.loc[index_num,"week4MaxBndQty"]=0
#             ndf.loc[index_num,"week4MinBndQty"]=0
#             ndf.loc[index_num,"week2AvgKind1Qty"]=0
#             ndf.loc[index_num,"week2MaxKind1Qty"]=0
#             ndf.loc[index_num,"week2MinKind1Qty"]=0
#             ndf.loc[index_num,"week3AvgKind1Qty"]=0
#             ndf.loc[index_num,"week3MaxKind1Qty"]=0
#             ndf.loc[index_num,"week3MinKind1Qty"]=0
#             ndf.loc[index_num,"week4AvgKind1Qty"]=0
#             ndf.loc[index_num,"week4MaxKind1Qty"]=0
#             ndf.loc[index_num,"week4MinKind1Qty"]=0
#             ndf.loc[index_num,"week2AvgKind2Qty"]=0
#             ndf.loc[index_num,"week2MaxKind2Qty"]=0
#             ndf.loc[index_num,"week2MinKind2Qty"]=0
#             ndf.loc[index_num,"week3AvgKind2Qty"]=0
#             ndf.loc[index_num,"week3MaxKind2Qty"]=0
#             ndf.loc[index_num,"week3MinKind2Qty"]=0
#             ndf.loc[index_num,"week4AvgKind2Qty"]=0
#             ndf.loc[index_num,"week4MaxKind2Qty"]=0
#             ndf.loc[index_num,"week4MinKind2Qty"]=0
#             ndf.loc[index_num,"week2AvgKind3Qty"]=0
#             ndf.loc[index_num,"week2MaxKind3Qty"]=0
#             ndf.loc[index_num,"week2MinKind3Qty"]=0
#             ndf.loc[index_num,"week3AvgKind3Qty"]=0
#             ndf.loc[index_num,"week3MaxKind3Qty"]=0
#             ndf.loc[index_num,"week3MinKind3Qty"]=0
#             ndf.loc[index_num,"week4AvgKind3Qty"]=0
#             ndf.loc[index_num,"week4MaxKind3Qty"]=0
#             ndf.loc[index_num,"week4MinKind3Qty"]=0
#             ndf.loc[index_num,"week2AvgKind4Qty"]=0
#             ndf.loc[index_num,"week2MaxKind4Qty"]=0
#             ndf.loc[index_num,"week2MinKind4Qty"]=0
#             ndf.loc[index_num,"week3AvgKind4Qty"]=0
#             ndf.loc[index_num,"week3MaxKind4Qty"]=0
#             ndf.loc[index_num,"week3MinKind4Qty"]=0
#             ndf.loc[index_num,"week4AvgKind4Qty"]=0
#             ndf.loc[index_num,"week4MaxKind4Qty"]=0
#             ndf.loc[index_num,"week4MinKind4Qty"]=0
#         else :
#             ndf.loc[index_num,"qty1"]=df_plu.iloc[j-1,i]
#             ndf.loc[index_num,"qty2"]=df_plu.iloc[j-2,i]
#             ndf.loc[index_num,"qty3"]=df_plu.iloc[j-3,i]
#             ndf.loc[index_num,"qty4"]=df_plu.iloc[j-4,i]
#             ndf.loc[index_num,"qty5"]=df_plu.iloc[j-5,i]
#             ndf.loc[index_num,"qty6"]=df_plu.iloc[j-6,i]
#             ndf.loc[index_num,"qty7"]=df_plu.iloc[j-7,i]
#             ndf.loc[index_num,"bndqty1"]=df_bnd.loc[sldatime_list[j-1],str(bndno)]
#             ndf.loc[index_num,"bndqty2"]=df_bnd.loc[sldatime_list[j-2],str(bndno)]
#             ndf.loc[index_num,"bndqty3"]=df_bnd.loc[sldatime_list[j-3],str(bndno)]
#             ndf.loc[index_num,"bndqty4"]=df_bnd.loc[sldatime_list[j-4],str(bndno)]
#             ndf.loc[index_num,"bndqty5"]=df_bnd.loc[sldatime_list[j-5],str(bndno)]
#             ndf.loc[index_num,"bndqty6"]=df_bnd.loc[sldatime_list[j-6],str(bndno)]
#             ndf.loc[index_num,"bndqty7"]=df_bnd.loc[sldatime_list[j-7],str(bndno)]
#             ndf.loc[index_num,"kind1qty1"]=df_kind1.loc[sldatime_list[j-1],str(kind1)]
#             ndf.loc[index_num,"kind1qty2"]=df_kind1.loc[sldatime_list[j-2],str(kind1)]
#             ndf.loc[index_num,"kind1qty3"]=df_kind1.loc[sldatime_list[j-3],str(kind1)]
#             ndf.loc[index_num,"kind1qty4"]=df_kind1.loc[sldatime_list[j-4],str(kind1)]
#             ndf.loc[index_num,"kind1qty5"]=df_kind1.loc[sldatime_list[j-5],str(kind1)]
#             ndf.loc[index_num,"kind1qty6"]=df_kind1.loc[sldatime_list[j-6],str(kind1)]
#             ndf.loc[index_num,"kind1qty7"]=df_kind1.loc[sldatime_list[j-7],str(kind1)]
#             ndf.loc[index_num,"kind2qty1"]=df_kind2.loc[sldatime_list[j-1],str(kind2)]
#             ndf.loc[index_num,"kind2qty2"]=df_kind2.loc[sldatime_list[j-2],str(kind2)]
#             ndf.loc[index_num,"kind2qty3"]=df_kind2.loc[sldatime_list[j-3],str(kind2)]
#             ndf.loc[index_num,"kind2qty4"]=df_kind2.loc[sldatime_list[j-4],str(kind2)]
#             ndf.loc[index_num,"kind2qty5"]=df_kind2.loc[sldatime_list[j-5],str(kind2)]
#             ndf.loc[index_num,"kind2qty6"]=df_kind2.loc[sldatime_list[j-6],str(kind2)]
#             ndf.loc[index_num,"kind2qty7"]=df_kind2.loc[sldatime_list[j-7],str(kind2)]
#             ndf.loc[index_num,"kind3qty1"]=df_kind3.loc[sldatime_list[j-1],str(kind3)]
#             ndf.loc[index_num,"kind3qty2"]=df_kind3.loc[sldatime_list[j-2],str(kind3)]
#             ndf.loc[index_num,"kind3qty3"]=df_kind3.loc[sldatime_list[j-3],str(kind3)]
#             ndf.loc[index_num,"kind3qty4"]=df_kind3.loc[sldatime_list[j-4],str(kind3)]
#             ndf.loc[index_num,"kind3qty5"]=df_kind3.loc[sldatime_list[j-5],str(kind3)]
#             ndf.loc[index_num,"kind3qty6"]=df_kind3.loc[sldatime_list[j-6],str(kind3)]
#             ndf.loc[index_num,"kind3qty7"]=df_kind3.loc[sldatime_list[j-7],str(kind3)]
#             ndf.loc[index_num,"kind4qty1"]=df_kind4.loc[sldatime_list[j-1],str(kind4)]
#             ndf.loc[index_num,"kind4qty2"]=df_kind4.loc[sldatime_list[j-2],str(kind4)]
#             ndf.loc[index_num,"kind4qty3"]=df_kind4.loc[sldatime_list[j-3],str(kind4)]
#             ndf.loc[index_num,"kind4qty4"]=df_kind4.loc[sldatime_list[j-4],str(kind4)]
#             ndf.loc[index_num,"kind4qty5"]=df_kind4.loc[sldatime_list[j-5],str(kind4)]
#             ndf.loc[index_num,"kind4qty6"]=df_kind4.loc[sldatime_list[j-6],str(kind4)]
#             ndf.loc[index_num,"kind4qty7"]=df_kind4.loc[sldatime_list[j-7],str(kind4)]
#             if j <= 14:
#                 ndf.loc[index_num,"week3AvgQty"]=0
#                 ndf.loc[index_num,"week3MaxQty"]=0
#                 ndf.loc[index_num,"week3MinQty"]=0
#                 ndf.loc[index_num,"week4AvgQty"]=0
#                 ndf.loc[index_num,"week4MaxQty"]=0
#                 ndf.loc[index_num,"week4MinQty"]=0
#                 ndf.loc[index_num,"week3AvgBndQty"]=0
#                 ndf.loc[index_num,"week3MaxBndQty"]=0
#                 ndf.loc[index_num,"week3MinBndQty"]=0
#                 ndf.loc[index_num,"week4AvgBndQty"]=0
#                 ndf.loc[index_num,"week4MaxBndQty"]=0
#                 ndf.loc[index_num,"week4MinBndQty"]=0
#                 ndf.loc[index_num,"week3AvgKind1Qty"]=0
#                 ndf.loc[index_num,"week3MaxKind1Qty"]=0
#                 ndf.loc[index_num,"week3MinKind1Qty"]=0
#                 ndf.loc[index_num,"week4AvgKind1Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind1Qty"]=0
#                 ndf.loc[index_num,"week4MinKind1Qty"]=0
#                 ndf.loc[index_num,"week3AvgKind2Qty"]=0
#                 ndf.loc[index_num,"week3MaxKind2Qty"]=0
#                 ndf.loc[index_num,"week3MinKind2Qty"]=0
#                 ndf.loc[index_num,"week4AvgKind2Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind2Qty"]=0
#                 ndf.loc[index_num,"week4MinKind2Qty"]=0
#                 ndf.loc[index_num,"week3AvgKind3Qty"]=0
#                 ndf.loc[index_num,"week3MaxKind3Qty"]=0
#                 ndf.loc[index_num,"week3MinKind3Qty"]=0
#                 ndf.loc[index_num,"week4AvgKind3Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind3Qty"]=0
#                 ndf.loc[index_num,"week4MinKind3Qty"]=0
#                 ndf.loc[index_num,"week3AvgKind4Qty"]=0
#                 ndf.loc[index_num,"week3MaxKind4Qty"]=0
#                 ndf.loc[index_num,"week3MinKind4Qty"]=0
#                 ndf.loc[index_num,"week4AvgKind4Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind4Qty"]=0
#                 ndf.loc[index_num,"week4MinKind4Qty"]=0
#                 m = 8
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= j:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week2AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week2MaxQty"]=max_qty
#                 ndf.loc[index_num,"week2MinQty"]=min_qty
#                 ndf.loc[index_num,"week2AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week2MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week2MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week2AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week2MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week2MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week2AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week2MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week2MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week2AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week2MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week2MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week2AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week2MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week2MinKind4Qty"]=min_kind4qty
#             elif j <= 21:
#                 ndf.loc[index_num,"week4AvgQty"]=0
#                 ndf.loc[index_num,"week4MaxQty"]=0
#                 ndf.loc[index_num,"week4MinQty"]=0
#                 ndf.loc[index_num,"week4AvgBndQty"]=0
#                 ndf.loc[index_num,"week4MaxBndQty"]=0
#                 ndf.loc[index_num,"week4MinBndQty"]=0
#                 ndf.loc[index_num,"week4AvgKind1Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind1Qty"]=0
#                 ndf.loc[index_num,"week4MinKind1Qty"]=0
#                 ndf.loc[index_num,"week4AvgKind2Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind2Qty"]=0
#                 ndf.loc[index_num,"week4MinKind2Qty"]=0
#                 ndf.loc[index_num,"week4AvgKind3Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind3Qty"]=0
#                 ndf.loc[index_num,"week4MinKind3Qty"]=0
#                 ndf.loc[index_num,"week4AvgKind4Qty"]=0
#                 ndf.loc[index_num,"week4MaxKind4Qty"]=0
#                 ndf.loc[index_num,"week4MinKind4Qty"]=0
#                 m = 8
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= 14:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week2AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week2MaxQty"]=max_qty
#                 ndf.loc[index_num,"week2MinQty"]=min_qty
#                 ndf.loc[index_num,"week2AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week2MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week2MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week2AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week2MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week2MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week2AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week2MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week2MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week2AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week2MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week2MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week2AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week2MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week2MinKind4Qty"]=min_kind4qty
#                 m = 14
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= j:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week3AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week3MaxQty"]=max_qty
#                 ndf.loc[index_num,"week3MinQty"]=min_qty
#                 ndf.loc[index_num,"week3AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week3MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week3MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week3AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week3MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week3MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week3AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week3MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week3MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week3AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week3MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week3MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week3AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week3MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week3MinKind4Qty"]=min_kind4qty
#             elif j <= 28:
#                 m = 8
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= 14:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week2AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week2MaxQty"]=max_qty
#                 ndf.loc[index_num,"week2MinQty"]=min_qty
#                 ndf.loc[index_num,"week2AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week2MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week2MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week2AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week2MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week2MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week2AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week2MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week2MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week2AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week2MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week2MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week2AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week2MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week2MinKind4Qty"]=min_kind4qty
#                 m = 14
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= 21:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week3AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week3MaxQty"]=max_qty
#                 ndf.loc[index_num,"week3MinQty"]=min_qty
#                 ndf.loc[index_num,"week3AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week3MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week3MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week3AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week3MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week3MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week3AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week3MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week3MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week3AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week3MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week3MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week3AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week3MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week3MinKind4Qty"]=min_kind4qty
#                 m = 21
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= j:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week4AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week4MaxQty"]=max_qty
#                 ndf.loc[index_num,"week4MinQty"]=min_qty
#                 ndf.loc[index_num,"week4AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week4MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week4MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week4AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week4MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week4MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week4AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week4MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week4MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week4AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week4MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week4MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week4AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week4MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week4MinKind4Qty"]=min_kind4qty
#             else:
#                 m = 8
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= 14:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week2AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week2MaxQty"]=max_qty
#                 ndf.loc[index_num,"week2MinQty"]=min_qty
#                 ndf.loc[index_num,"week2AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week2MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week2MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week2AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week2MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week2MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week2AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week2MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week2MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week2AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week2MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week2MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week2AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week2MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week2MinKind4Qty"]=min_kind4qty
#                 m = 14
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= 21:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week3AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week3MaxQty"]=max_qty
#                 ndf.loc[index_num,"week3MinQty"]=min_qty
#                 ndf.loc[index_num,"week3AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week3MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week3MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week3AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week3MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week3MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week3AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week3MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week3MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week3AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week3MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week3MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week3AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week3MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week3MinKind4Qty"]=min_kind4qty
#                 m = 21
#                 sum_qty = 0
#                 sum_bndqty = 0
#                 sum_kind1qty = 0
#                 sum_kind2qty = 0
#                 sum_kind3qty = 0
#                 sum_kind4qty = 0
#                 max_qty = 0
#                 max_bndqty = 0
#                 max_kind1qty = 0
#                 max_kind2qty = 0
#                 max_kind3qty = 0
#                 max_kind4qty = 0
#                 min_qty = 1000
#                 min_bndqty = 1000
#                 min_kind1qty = 1000
#                 min_kind2qty = 1000
#                 min_kind3qty = 1000
#                 min_kind4qty = 1000
#                 while m <= 28:
#                     qty = df_plu.iloc[j-m,i]
#                     bndqty = df_bnd.loc[sldatime_list[j-m],str(bndno)]
#                     kind1qty = df_kind1.loc[sldatime_list[j-m],str(kind1)]
#                     kind2qty = df_kind2.loc[sldatime_list[j-m],str(kind2)]
#                     kind3qty = df_kind3.loc[sldatime_list[j-m],str(kind3)]
#                     kind4qty = df_kind4.loc[sldatime_list[j-m],str(kind4)]
#                     sum_qty = sum_qty + qty
#                     sum_bndqty = sum_bndqty + bndqty
#                     sum_kind1qty = sum_kind1qty + kind1qty
#                     sum_kind2qty = sum_kind2qty + kind2qty
#                     sum_kind3qty = sum_kind3qty + kind3qty
#                     sum_kind4qty = sum_kind4qty + kind4qty
#                     max_qty = np.max([max_qty, qty])
#                     max_bndqty = np.max([max_bndqty, bndqty])
#                     max_kind1qty = np.max([max_kind1qty, kind1qty])
#                     max_kind2qty = np.max([max_kind2qty, kind2qty])
#                     max_kind3qty = np.max([max_kind3qty, kind3qty])
#                     max_kind4qty = np.max([max_kind4qty, kind4qty])
#                     min_qty = np.min([min_qty, qty])
#                     min_bndqty = np.min([min_bndqty, bndqty])
#                     min_kind1qty = np.min([min_kind1qty, kind1qty])
#                     min_kind2qty = np.min([min_kind2qty, kind2qty])
#                     min_kind3qty = np.min([min_kind3qty, kind3qty])
#                     min_kind4qty = np.min([min_kind4qty, kind4qty])
#                     m = m + 1
#                 ndf.loc[index_num,"week4AvgQty"]=sum_qty/7
#                 ndf.loc[index_num,"week4MaxQty"]=max_qty
#                 ndf.loc[index_num,"week4MinQty"]=min_qty
#                 ndf.loc[index_num,"week4AvgBndQty"]=sum_bndqty/7
#                 ndf.loc[index_num,"week4MaxBndQty"]=max_bndqty
#                 ndf.loc[index_num,"week4MinBndQty"]=min_bndqty
#                 ndf.loc[index_num,"week4AvgKind1Qty"]=sum_kind1qty/7
#                 ndf.loc[index_num,"week4MaxKind1Qty"]=max_kind1qty
#                 ndf.loc[index_num,"week4MinKind1Qty"]=min_kind1qty
#                 ndf.loc[index_num,"week4AvgKind2Qty"]=sum_kind2qty/7
#                 ndf.loc[index_num,"week4MaxKind2Qty"]=max_kind2qty
#                 ndf.loc[index_num,"week4MinKind2Qty"]=min_kind2qty
#                 ndf.loc[index_num,"week4AvgKind3Qty"]=sum_kind3qty/7
#                 ndf.loc[index_num,"week4MaxKind3Qty"]=max_kind3qty
#                 ndf.loc[index_num,"week4MinKind3Qty"]=min_kind3qty
#                 ndf.loc[index_num,"week4AvgKind4Qty"]=sum_kind4qty/7
#                 ndf.loc[index_num,"week4MaxKind4Qty"]=max_kind4qty
#                 ndf.loc[index_num,"week4MinKind4Qty"]=min_kind4qty
ndf


# In[247]:


# def getQty(o, num, dataframe):
#     sldatime = datetime.datetime(int(o.sldatime[0:4]),int(o.sldatime[5:7]),int(o.sldatime[8:10]))
#     searchTime = sldatime - datetime.timedelta(num)
#     if searchTime < datetime.datetime(2016,2,1):
#         return 0
#     else:
#         searchdf = dataframe.loc[dataframe["pluno"] == o.pluno]
#         searchdf = searchdf.reset_index(drop=True)
#         qty = 0
#         for i in range(searchdf.shape[0]):
#             if searchdf.loc[i,"sldatime"] == searchTime.strftime("%Y-%m-%d"):
#                 qty = qty + searchdf.loc[i,"qty"]
#         return qty

    
# def getBndQty(o, num, dataframe):
#     sldatime = datetime.datetime(int(o.sldatime[0:4]),int(o.sldatime[5:7]),int(o.sldatime[8:10]))
#     searchTime = sldatime - datetime.timedelta(num)
#     if searchTime < datetime.datetime(2016,2,1):
#         return 0
#     else:
#         searchdf = dataframe.loc[dataframe["bndno"] == o.bndno]
#         searchdf = searchdf.reset_index(drop=True)
#         qty = 0
#         for i in range(searchdf.shape[0]):
#             if searchdf.loc[i,"sldatime"] == searchTime.strftime("%Y-%m-%d"):
#                 qty = qty + searchdf.loc[i,"qty"]
#         return qty
    
    
# def getKind1Qty(o, num, dataframe):
#     sldatime = datetime.datetime(int(o.sldatime[0:4]),int(o.sldatime[5:7]),int(o.sldatime[8:10]))
#     searchTime = sldatime - datetime.timedelta(num)
#     if searchTime < datetime.datetime(2016,2,1):
#         return 0
#     else:
#         searchdf = dataframe.loc[dataframe["kind1"] == o.kind1]
#         searchdf = searchdf.reset_index(drop=True)
#         qty = 0
#         for i in range(searchdf.shape[0]):
#             if searchdf.loc[i,"sldatime"] == searchTime.strftime("%Y-%m-%d"):
#                 qty = qty + searchdf.loc[i,"qty"]
#         return qty
    
    
# def getKind2Qty(o, num, dataframe):
#     sldatime = datetime.datetime(int(o.sldatime[0:4]),int(o.sldatime[5:7]),int(o.sldatime[8:10]))
#     searchTime = sldatime - datetime.timedelta(num)
#     if searchTime < datetime.datetime(2016,2,1):
#         return 0
#     else:
#         searchdf = dataframe.loc[dataframe["kind2"] == o.kind2]
#         searchdf = searchdf.reset_index(drop=True)
#         qty = 0
#         for i in range(searchdf.shape[0]):
#             if searchdf.loc[i,"sldatime"] == searchTime.strftime("%Y-%m-%d"):
#                 qty = qty + searchdf.loc[i,"qty"]
#         return qty
    
    
# def getKind3Qty(o, num, dataframe):
#     sldatime = datetime.datetime(int(o.sldatime[0:4]),int(o.sldatime[5:7]),int(o.sldatime[8:10]))
#     searchTime = sldatime - datetime.timedelta(num)
#     if searchTime < datetime.datetime(2016,2,1):
#         return 0
#     else:
#         searchdf = dataframe.loc[dataframe["kind3"] == o.kind3]
#         searchdf = searchdf.reset_index(drop=True)
#         qty = 0
#         for i in range(searchdf.shape[0]):
#             if searchdf.loc[i,"sldatime"] == searchTime.strftime("%Y-%m-%d"):
#                 qty = qty + searchdf.loc[i,"qty"]
#         return qty
    
    
# def getKind4Qty(o, num, dataframe):
#     sldatime = datetime.datetime(int(o.sldatime[0:4]),int(o.sldatime[5:7]),int(o.sldatime[8:10]))
#     searchTime = sldatime - datetime.timedelta(num)
#     if searchTime < datetime.datetime(2016,2,1):
#         return 0
#     else:
#         searchdf = dataframe.loc[dataframe["kind4"] == o.kind4]
#         searchdf = searchdf.reset_index(drop=True)
#         qty = 0
#         for i in range(searchdf.shape[0]):
#             if searchdf.loc[i,"sldatime"] == searchTime.strftime("%Y-%m-%d"):
#                 qty = qty + searchdf.loc[i,"qty"]
#         return qty


# In[250]:


# for i in range(ndf.shape[0]):
#     print(i)
#     ndf.loc[i,"qty1"] = getQty(ndf.loc[i,:], 1, df)
#     ndf.loc[i,"qty2"] = getQty(ndf.loc[i,:], 2, df)
#     ndf.loc[i,"qty3"] = getQty(ndf.loc[i,:], 3, df)
#     ndf.loc[i,"qty4"] = getQty(ndf.loc[i,:], 4, df)
#     ndf.loc[i,"qty5"] = getQty(ndf.loc[i,:], 5, df)
#     ndf.loc[i,"qty6"] = getQty(ndf.loc[i,:], 6, df)
#     ndf.loc[i,"qty7"] = getQty(ndf.loc[i,:], 7, df)
#     ndf.loc[i,"bndqty1"] = getBndQty(ndf.loc[i,:], 1, df)
#     ndf.loc[i,"bndqty2"] = getBndQty(ndf.loc[i,:], 2, df)
#     ndf.loc[i,"bndqty3"] = getBndQty(ndf.loc[i,:], 3, df)
#     ndf.loc[i,"bndqty4"] = getBndQty(ndf.loc[i,:], 4, df)
#     ndf.loc[i,"bndqty5"] = getBndQty(ndf.loc[i,:], 5, df)
#     ndf.loc[i,"bndqty6"] = getBndQty(ndf.loc[i,:], 6, df)
#     ndf.loc[i,"bndqty7"] = getBndQty(ndf.loc[i,:], 7, df)
#     ndf.loc[i,"kind1qty1"] = getKind1Qty(ndf.loc[i,:], 1, df)
#     ndf.loc[i,"kind1qty2"] = getKind1Qty(ndf.loc[i,:], 2, df)
#     ndf.loc[i,"kind1qty3"] = getKind1Qty(ndf.loc[i,:], 3, df)
#     ndf.loc[i,"kind1qty4"] = getKind1Qty(ndf.loc[i,:], 4, df)
#     ndf.loc[i,"kind1qty5"] = getKind1Qty(ndf.loc[i,:], 5, df)
#     ndf.loc[i,"kind1qty6"] = getKind1Qty(ndf.loc[i,:], 6, df)
#     ndf.loc[i,"kind1qty7"] = getKind1Qty(ndf.loc[i,:], 7, df)
    
#     ndf.loc[i,"kind2qty1"] = getKind2Qty(ndf.loc[i,:], 1, df)
#     ndf.loc[i,"kind2qty2"] = getKind2Qty(ndf.loc[i,:], 2, df)
#     ndf.loc[i,"kind2qty3"] = getKind2Qty(ndf.loc[i,:], 3, df)
#     ndf.loc[i,"kind2qty4"] = getKind2Qty(ndf.loc[i,:], 4, df)
#     ndf.loc[i,"kind2qty5"] = getKind2Qty(ndf.loc[i,:], 5, df)
#     ndf.loc[i,"kind2qty6"] = getKind2Qty(ndf.loc[i,:], 6, df)
#     ndf.loc[i,"kind2qty7"] = getKind2Qty(ndf.loc[i,:], 7, df)
    
#     ndf.loc[i,"kind3qty1"] = getKind3Qty(ndf.loc[i,:], 1, df)
#     ndf.loc[i,"kind3qty2"] = getKind3Qty(ndf.loc[i,:], 2, df)
#     ndf.loc[i,"kind3qty3"] = getKind3Qty(ndf.loc[i,:], 3, df)
#     ndf.loc[i,"kind3qty4"] = getKind3Qty(ndf.loc[i,:], 4, df)
#     ndf.loc[i,"kind3qty5"] = getKind3Qty(ndf.loc[i,:], 5, df)
#     ndf.loc[i,"kind3qty6"] = getKind3Qty(ndf.loc[i,:], 6, df)
#     ndf.loc[i,"kind3qty7"] = getKind3Qty(ndf.loc[i,:], 7, df)
    
#     ndf.loc[i,"kind4qty1"] = getKind4Qty(ndf.loc[i,:], 1, df)
#     ndf.loc[i,"kind4qty2"] = getKind4Qty(ndf.loc[i,:], 2, df)
#     ndf.loc[i,"kind4qty3"] = getKind4Qty(ndf.loc[i,:], 3, df)
#     ndf.loc[i,"kind4qty4"] = getKind4Qty(ndf.loc[i,:], 4, df)
#     ndf.loc[i,"kind4qty5"] = getKind4Qty(ndf.loc[i,:], 5, df)
#     ndf.loc[i,"kind4qty6"] = getKind4Qty(ndf.loc[i,:], 6, df)
#     ndf.loc[i,"kind4qty7"] = getKind4Qty(ndf.loc[i,:], 7, df)
    
#     qty8 = getQty(ndf.loc[i,:], 8, df)
#     qty9 = getQty(ndf.loc[i,:], 9, df)
#     qty10 = getQty(ndf.loc[i,:], 10, df)
#     qty11 = getQty(ndf.loc[i,:], 11, df)
#     qty12 = getQty(ndf.loc[i,:], 12, df)
#     qty13 = getQty(ndf.loc[i,:], 13, df)
#     qty14 = getQty(ndf.loc[i,:], 14, df)
#     qty15 = getQty(ndf.loc[i,:], 15, df)
#     qty16 = getQty(ndf.loc[i,:], 16, df)
#     qty17 = getQty(ndf.loc[i,:], 17, df)
#     qty18 = getQty(ndf.loc[i,:], 18, df)
#     qty19 = getQty(ndf.loc[i,:], 19, df)
#     qty20 = getQty(ndf.loc[i,:], 20, df)
#     qty21 = getQty(ndf.loc[i,:], 21, df)
#     qty22 = getQty(ndf.loc[i,:], 22, df)
#     qty23 = getQty(ndf.loc[i,:], 23, df)
#     qty24 = getQty(ndf.loc[i,:], 24, df)
#     qty25 = getQty(ndf.loc[i,:], 25, df)
#     qty26 = getQty(ndf.loc[i,:], 26, df)
#     qty27 = getQty(ndf.loc[i,:], 27, df)
#     qty28 = getQty(ndf.loc[i,:], 28, df)
#     ndf.loc[i,"week2AvgQty"] = np.mean([qty8,qty9,qty10,qty11,qty12,qty13,qty14])
#     ndf.loc[i,"week2MaxQty"] = np.max([qty8,qty9,qty10,qty11,qty12,qty13,qty14])
#     ndf.loc[i,"week2MinQty"] = np.min([qty8,qty9,qty10,qty11,qty12,qty13,qty14])
#     ndf.loc[i,"week3AvgQty"] = np.mean([qty15,qty16,qty17,qty18,qty19,qty20,qty21])
#     ndf.loc[i,"week3MaxQty"] = np.max([qty15,qty16,qty17,qty18,qty19,qty20,qty21])
#     ndf.loc[i,"week3MinQty"] = np.min([qty15,qty16,qty17,qty18,qty19,qty20,qty21])
#     ndf.loc[i,"week4AvgQty"] = np.mean([qty22,qty23,qty24,qty25,qty26,qty27,qty28])
#     ndf.loc[i,"week4MaxQty"] = np.max([qty22,qty23,qty24,qty25,qty26,qty27,qty28])
#     ndf.loc[i,"week4MinQty"] = np.min([qty22,qty23,qty24,qty25,qty26,qty27,qty28])
#     bndqty8 = getBndQty(ndf.loc[i,:], 8, df)
#     bndqty9 = getBndQty(ndf.loc[i,:], 9, df)
#     bndqty10 = getBndQty(ndf.loc[i,:], 10, df)
#     bndqty11 = getBndQty(ndf.loc[i,:], 11, df)
#     bndqty12 = getBndQty(ndf.loc[i,:], 12, df)
#     bndqty13 = getBndQty(ndf.loc[i,:], 13, df)
#     bndqty14 = getBndQty(ndf.loc[i,:], 14, df)
#     bndqty15 = getBndQty(ndf.loc[i,:], 15, df)
#     bndqty16 = getBndQty(ndf.loc[i,:], 16, df)
#     bndqty17 = getBndQty(ndf.loc[i,:], 17, df)
#     bndqty18 = getBndQty(ndf.loc[i,:], 18, df)
#     bndqty19 = getBndQty(ndf.loc[i,:], 19, df)
#     bndqty20 = getBndQty(ndf.loc[i,:], 20, df)
#     bndqty21 = getBndQty(ndf.loc[i,:], 21, df)
#     bndqty22 = getBndQty(ndf.loc[i,:], 22, df)
#     bndqty23 = getBndQty(ndf.loc[i,:], 23, df)
#     bndqty24 = getBndQty(ndf.loc[i,:], 24, df)
#     bndqty25 = getBndQty(ndf.loc[i,:], 25, df)
#     bndqty26 = getBndQty(ndf.loc[i,:], 26, df)
#     bndqty27 = getBndQty(ndf.loc[i,:], 27, df)
#     bndqty28 = getBndQty(ndf.loc[i,:], 28, df)
#     ndf.loc[i,"week2AvgBndQty"] = np.mean([bndqty8,bndqty9,bndqty10,bndqty11,bndqty12,bndqty13,bndqty14])
#     ndf.loc[i,"week2MaxBndQty"] = np.max([bndqty8,bndqty9,bndqty10,bndqty11,bndqty12,bndqty13,bndqty14])
#     ndf.loc[i,"week2MinBndQty"] = np.min([bndqty8,bndqty9,bndqty10,bndqty11,bndqty12,bndqty13,bndqty14])
#     ndf.loc[i,"week3AvgBndQty"] = np.mean([bndqty15,bndqty16,bndqty17,bndqty18,bndqty19,bndqty20,bndqty21])
#     ndf.loc[i,"week3MaxBndQty"] = np.max([bndqty15,bndqty16,bndqty17,bndqty18,bndqty19,bndqty20,bndqty21])
#     ndf.loc[i,"week3MinBndQty"] = np.min([bndqty15,bndqty16,bndqty17,bndqty18,bndqty19,bndqty20,bndqty21])
#     ndf.loc[i,"week4AvgBndQty"] = np.mean([bndqty22,bndqty23,bndqty24,bndqty25,bndqty26,bndqty27,bndqty28])
#     ndf.loc[i,"week4MaxBndQty"] = np.max([bndqty22,bndqty23,bndqty24,bndqty25,bndqty26,bndqty27,bndqty28])
#     ndf.loc[i,"week4MinBndQty"] = np.min([bndqty22,bndqty23,bndqty24,bndqty25,bndqty26,bndqty27,bndqty28])
#     kind1qty8 = getKind1Qty(ndf.loc[i,:], 8, df)
#     kind1qty9 = getKind1Qty(ndf.loc[i,:], 9, df)
#     kind1qty10 = getKind1Qty(ndf.loc[i,:], 10, df)
#     kind1qty11 = getKind1Qty(ndf.loc[i,:], 11, df)
#     kind1qty12 = getKind1Qty(ndf.loc[i,:], 12, df)
#     kind1qty13 = getKind1Qty(ndf.loc[i,:], 13, df)
#     kind1qty14 = getKind1Qty(ndf.loc[i,:], 14, df)
#     kind1qty15 = getKind1Qty(ndf.loc[i,:], 15, df)
#     kind1qty16 = getKind1Qty(ndf.loc[i,:], 16, df)
#     kind1qty17 = getKind1Qty(ndf.loc[i,:], 17, df)
#     kind1qty18 = getKind1Qty(ndf.loc[i,:], 18, df)
#     kind1qty19 = getKind1Qty(ndf.loc[i,:], 19, df)
#     kind1qty20 = getKind1Qty(ndf.loc[i,:], 20, df)
#     kind1qty21 = getKind1Qty(ndf.loc[i,:], 21, df)
#     kind1qty22 = getKind1Qty(ndf.loc[i,:], 22, df)
#     kind1qty23 = getKind1Qty(ndf.loc[i,:], 23, df)
#     kind1qty24 = getKind1Qty(ndf.loc[i,:], 24, df)
#     kind1qty25 = getKind1Qty(ndf.loc[i,:], 25, df)
#     kind1qty26 = getKind1Qty(ndf.loc[i,:], 26, df)
#     kind1qty27 = getKind1Qty(ndf.loc[i,:], 27, df)
#     kind1qty28 = getKind1Qty(ndf.loc[i,:], 28, df)
    
#     kind2qty8 = getKind2Qty(ndf.loc[i,:], 8, df)
#     kind2qty9 = getKind2Qty(ndf.loc[i,:], 9, df)
#     kind2qty10 = getKind2Qty(ndf.loc[i,:], 10, df)
#     kind2qty11 = getKind2Qty(ndf.loc[i,:], 11, df)
#     kind2qty12 = getKind2Qty(ndf.loc[i,:], 12, df)
#     kind2qty13 = getKind2Qty(ndf.loc[i,:], 13, df)
#     kind2qty14 = getKind2Qty(ndf.loc[i,:], 14, df)
#     kind2qty15 = getKind2Qty(ndf.loc[i,:], 15, df)
#     kind2qty16 = getKind2Qty(ndf.loc[i,:], 16, df)
#     kind2qty17 = getKind2Qty(ndf.loc[i,:], 17, df)
#     kind2qty18 = getKind2Qty(ndf.loc[i,:], 18, df)
#     kind2qty19 = getKind2Qty(ndf.loc[i,:], 19, df)
#     kind2qty20 = getKind2Qty(ndf.loc[i,:], 20, df)
#     kind2qty21 = getKind2Qty(ndf.loc[i,:], 21, df)
#     kind2qty22 = getKind2Qty(ndf.loc[i,:], 22, df)
#     kind2qty23 = getKind2Qty(ndf.loc[i,:], 23, df)
#     kind2qty24 = getKind2Qty(ndf.loc[i,:], 24, df)
#     kind2qty25 = getKind2Qty(ndf.loc[i,:], 25, df)
#     kind2qty26 = getKind2Qty(ndf.loc[i,:], 26, df)
#     kind2qty27 = getKind2Qty(ndf.loc[i,:], 27, df)
#     kind2qty28 = getKind2Qty(ndf.loc[i,:], 28, df)
    
#     kind3qty8 = getKind3Qty(ndf.loc[i,:], 8, df)
#     kind3qty9 = getKind3Qty(ndf.loc[i,:], 9, df)
#     kind3qty10 = getKind3Qty(ndf.loc[i,:], 10, df)
#     kind3qty11 = getKind3Qty(ndf.loc[i,:], 11, df)
#     kind3qty12 = getKind3Qty(ndf.loc[i,:], 12, df)
#     kind3qty13 = getKind3Qty(ndf.loc[i,:], 13, df)
#     kind3qty14 = getKind3Qty(ndf.loc[i,:], 14, df)
#     kind3qty15 = getKind3Qty(ndf.loc[i,:], 15, df)
#     kind3qty16 = getKind3Qty(ndf.loc[i,:], 16, df)
#     kind3qty17 = getKind3Qty(ndf.loc[i,:], 17, df)
#     kind3qty18 = getKind3Qty(ndf.loc[i,:], 18, df)
#     kind3qty19 = getKind3Qty(ndf.loc[i,:], 19, df)
#     kind3qty20 = getKind3Qty(ndf.loc[i,:], 20, df)
#     kind3qty21 = getKind3Qty(ndf.loc[i,:], 21, df)
#     kind3qty22 = getKind3Qty(ndf.loc[i,:], 22, df)
#     kind3qty23 = getKind3Qty(ndf.loc[i,:], 23, df)
#     kind3qty24 = getKind3Qty(ndf.loc[i,:], 24, df)
#     kind3qty25 = getKind3Qty(ndf.loc[i,:], 25, df)
#     kind3qty26 = getKind3Qty(ndf.loc[i,:], 26, df)
#     kind3qty27 = getKind3Qty(ndf.loc[i,:], 27, df)
#     kind3qty28 = getKind3Qty(ndf.loc[i,:], 28, df)
    
#     kind4qty8 = getKind4Qty(ndf.loc[i,:], 8, df)
#     kind4qty9 = getKind4Qty(ndf.loc[i,:], 9, df)
#     kind4qty10 = getKind4Qty(ndf.loc[i,:], 10, df)
#     kind4qty11 = getKind4Qty(ndf.loc[i,:], 11, df)
#     kind4qty12 = getKind4Qty(ndf.loc[i,:], 12, df)
#     kind4qty13 = getKind4Qty(ndf.loc[i,:], 13, df)
#     kind4qty14 = getKind4Qty(ndf.loc[i,:], 14, df)
#     kind4qty15 = getKind4Qty(ndf.loc[i,:], 15, df)
#     kind4qty16 = getKind4Qty(ndf.loc[i,:], 16, df)
#     kind4qty17 = getKind4Qty(ndf.loc[i,:], 17, df)
#     kind4qty18 = getKind4Qty(ndf.loc[i,:], 18, df)
#     kind4qty19 = getKind4Qty(ndf.loc[i,:], 19, df)
#     kind4qty20 = getKind4Qty(ndf.loc[i,:], 20, df)
#     kind4qty21 = getKind4Qty(ndf.loc[i,:], 21, df)
#     kind4qty22 = getKind4Qty(ndf.loc[i,:], 22, df)
#     kind4qty23 = getKind4Qty(ndf.loc[i,:], 23, df)
#     kind4qty24 = getKind4Qty(ndf.loc[i,:], 24, df)
#     kind4qty25 = getKind4Qty(ndf.loc[i,:], 25, df)
#     kind4qty26 = getKind4Qty(ndf.loc[i,:], 26, df)
#     kind4qty27 = getKind4Qty(ndf.loc[i,:], 27, df)
#     kind4qty28 = getKind4Qty(ndf.loc[i,:], 28, df)
    
#     ndf.loc[i,"week2AvgKind1Qty"] = np.mean([kind1qty8,kind1qty9,kind1qty10,kind1qty11,kind1qty12,kind1qty13,kind1qty14])
#     ndf.loc[i,"week2MaxKind1Qty"] = np.max([kind1qty8,kind1qty9,kind1qty10,kind1qty11,kind1qty12,kind1qty13,kind1qty14])
#     ndf.loc[i,"week2MinKind1Qty"] = np.min([kind1qty8,kind1qty9,kind1qty10,kind1qty11,kind1qty12,kind1qty13,kind1qty14])
#     ndf.loc[i,"week3AvgKind1Qty"] = np.mean([kind1qty15,kind1qty16,kind1qty17,kind1qty18,kind1qty19,kind1qty20,kind1qty21])
#     ndf.loc[i,"week3MaxKind1Qty"] = np.max([kind1qty15,kind1qty16,kind1qty17,kind1qty18,kind1qty19,kind1qty20,kind1qty21])
#     ndf.loc[i,"week3MinKind1Qty"] = np.min([kind1qty15,kind1qty16,kind1qty17,kind1qty18,kind1qty19,kind1qty20,kind1qty21])
#     ndf.loc[i,"week4AvgKind1Qty"] = np.mean([kind1qty22,kind1qty23,kind1qty24,kind1qty25,kind1qty26,kind1qty27,kind1qty28])
#     ndf.loc[i,"week4MaxKind1Qty"] = np.max([kind1qty22,kind1qty23,kind1qty24,kind1qty25,kind1qty26,kind1qty27,kind1qty28])
#     ndf.loc[i,"week4MinKind1Qty"] = np.min([kind1qty22,kind1qty23,kind1qty24,kind1qty25,kind1qty26,kind1qty27,kind1qty28])
    
#     ndf.loc[i,"week2AvgKind2Qty"] = np.mean([kind2qty8,kind2qty9,kind2qty10,kind2qty11,kind2qty12,kind2qty13,kind2qty14])
#     ndf.loc[i,"week2MaxKind2Qty"] = np.max([kind2qty8,kind2qty9,kind2qty10,kind2qty11,kind2qty12,kind2qty13,kind2qty14])
#     ndf.loc[i,"week2MinKind2Qty"] = np.min([kind2qty8,kind2qty9,kind2qty10,kind2qty11,kind2qty12,kind2qty13,kind2qty14])
#     ndf.loc[i,"week3AvgKind2Qty"] = np.mean([kind2qty15,kind2qty16,kind2qty17,kind2qty18,kind2qty19,kind2qty20,kind2qty21])
#     ndf.loc[i,"week3MaxKind2Qty"] = np.max([kind2qty15,kind2qty16,kind2qty17,kind2qty18,kind2qty19,kind2qty20,kind2qty21])
#     ndf.loc[i,"week3MinKind2Qty"] = np.min([kind2qty15,kind2qty16,kind2qty17,kind2qty18,kind2qty19,kind2qty20,kind2qty21])
#     ndf.loc[i,"week4AvgKind2Qty"] = np.mean([kind2qty22,kind2qty23,kind2qty24,kind2qty25,kind2qty26,kind2qty27,kind2qty28])
#     ndf.loc[i,"week4MaxKind2Qty"] = np.max([kind2qty22,kind2qty23,kind2qty24,kind2qty25,kind2qty26,kind2qty27,kind2qty28])
#     ndf.loc[i,"week4MinKind2Qty"] = np.min([kind2qty22,kind2qty23,kind2qty24,kind2qty25,kind2qty26,kind2qty27,kind2qty28])
    
#     ndf.loc[i,"week2AvgKind3Qty"] = np.mean([kind3qty8,kind3qty9,kind3qty10,kind3qty11,kind3qty12,kind3qty13,kind3qty14])
#     ndf.loc[i,"week2MaxKind3Qty"] = np.max([kind3qty8,kind3qty9,kind3qty10,kind3qty11,kind3qty12,kind3qty13,kind3qty14])
#     ndf.loc[i,"week2MinKind3Qty"] = np.min([kind3qty8,kind3qty9,kind3qty10,kind3qty11,kind3qty12,kind3qty13,kind3qty14])
#     ndf.loc[i,"week3AvgKind3Qty"] = np.mean([kind3qty15,kind3qty16,kind3qty17,kind3qty18,kind3qty19,kind3qty20,kind3qty21])
#     ndf.loc[i,"week3MaxKind3Qty"] = np.max([kind3qty15,kind3qty16,kind3qty17,kind3qty18,kind3qty19,kind3qty20,kind3qty21])
#     ndf.loc[i,"week3MinKind3Qty"] = np.min([kind3qty15,kind3qty16,kind3qty17,kind3qty18,kind3qty19,kind3qty20,kind3qty21])
#     ndf.loc[i,"week4AvgKind3Qty"] = np.mean([kind3qty22,kind3qty23,kind3qty24,kind3qty25,kind3qty26,kind3qty27,kind3qty28])
#     ndf.loc[i,"week4MaxKind3Qty"] = np.max([kind3qty22,kind3qty23,kind3qty24,kind3qty25,kind3qty26,kind3qty27,kind3qty28])
#     ndf.loc[i,"week4MinKind3Qty"] = np.min([kind3qty22,kind3qty23,kind3qty24,kind3qty25,kind3qty26,kind3qty27,kind3qty28])
    
#     ndf.loc[i,"week2AvgKind4Qty"] = np.mean([kind4qty8,kind4qty9,kind4qty10,kind4qty11,kind4qty12,kind4qty13,kind4qty14])
#     ndf.loc[i,"week2MaxKind4Qty"] = np.max([kind4qty8,kind4qty9,kind4qty10,kind4qty11,kind4qty12,kind4qty13,kind4qty14])
#     ndf.loc[i,"week2MinKind4Qty"] = np.min([kind4qty8,kind4qty9,kind4qty10,kind4qty11,kind4qty12,kind4qty13,kind4qty14])
#     ndf.loc[i,"week3AvgKind4Qty"] = np.mean([kind4qty15,kind4qty16,kind4qty17,kind4qty18,kind4qty19,kind4qty20,kind4qty21])
#     ndf.loc[i,"week3MaxKind4Qty"] = np.max([kind4qty15,kind4qty16,kind4qty17,kind4qty18,kind4qty19,kind4qty20,kind4qty21])
#     ndf.loc[i,"week3MinKind4Qty"] = np.min([kind4qty15,kind4qty16,kind4qty17,kind4qty18,kind4qty19,kind4qty20,kind4qty21])
#     ndf.loc[i,"week4AvgKind4Qty"] = np.mean([kind4qty22,kind4qty23,kind4qty24,kind4qty25,kind4qty26,kind4qty27,kind4qty28])
#     ndf.loc[i,"week4MaxKind4Qty"] = np.max([kind4qty22,kind4qty23,kind4qty24,kind4qty25,kind4qty26,kind4qty27,kind4qty28])
#     ndf.loc[i,"week4MinKind4Qty"] = np.min([kind4qty22,kind4qty23,kind4qty24,kind4qty25,kind4qty26,kind4qty27,kind4qty28])
# ndf


# In[447]:


feature1


# In[471]:


feature2


# In[472]:


feature3


# In[473]:


feature4


# In[474]:


feature5


# In[475]:


feature6


# In[ ]:




