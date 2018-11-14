# @Time : 2018/11/14 下午8:12 
# @Author : Kaishun Zhang 
# @File : compare_others.py 
# @Function:
# 自己程序和别人的程序的区别，主要是以学习为主
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = np.random.rand(1000,3)
data = data[0::1,0:2:1]   # the third parameter is the step
print(data.shape) # (1000,2)

label = np.random.rand(1000)
# 将原始数据进行拆分
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size = 0.2,random_state = 23323)
print(train_data.shape)
print(test_data.shape)

x = np.ones(10)
y = np.zeros(10)
accuracy = accuracy_score(x,y) # 只是计算这个比值
print('accuracy',accuracy)