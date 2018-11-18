# @Time : 2018/11/18 下午12:15 
# @Author : Kaishun Zhang 
# @File : data_analysis.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
# 数据格式已改，需要修改代码
frame = pd.read_csv('watermelon3.csv')
data = frame.values
data = data[data[:,7].argsort()]
data_less = data[data[:,7] < 0.381]
data_more = data[data[:,7] > 0.381]
cate_counter = Counter(data[:,-1])
entroyD = 0
for key,value in cate_counter.items():
    entroyD += (-value / len(data) * np.log2(value / len(data)))

sub_entroy = 0
entroy_less = 0
cate_counter = Counter(data_less[:,-1])
for key,value in cate_counter.items():
    entroy_less += (-value / len(data_less) * np.log2(value / len(data_less)))
entroy_less = len(data_less) / len(data) * entroy_less

entroy_more = 0
cate_counter = Counter(data_more[:,-1])
for key,value in cate_counter.items():
    entroy_more += (-value / len(data_more) * np.log2(value / len(data_more)))
entroy_more = len(data_more) / len(data) * entroy_more

sub_entroy = entroy_more + entroy_less

print(entroyD - sub_entroy)