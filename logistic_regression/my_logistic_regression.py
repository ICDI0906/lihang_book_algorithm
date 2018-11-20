# @Time : 2018/11/19 上午9:27 
# @Author : Kaishun Zhang 
# @File : my_binary_perceptron.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
class BP(object):
    def __init__(self,learning_rate = 1e-5,max_iterator = 10):
        self.learning_rate = learning_rate
        self.max_iterator = max_iterator

    def train(self,train_data,train_label):
        self.w = np.zeros(train_data.shape[1] + 1)
        for iterator_i in range(self.max_iterator):
            for i,data in enumerate(train_data):
                data = np.append(data,1) #是种浅拷贝 ！！！
                tmp = np.dot(self.w,data.transpose())
                self.w += self.learning_rate * (data * train_label[i] - data * np.exp(tmp)/(1 + np.exp(tmp))) # there should be + 因为是最大化

    def predict(self,test):
        # print(self.w)
        test = np.append(test,1)
        return int(1 / (1 + np.exp(np.dot(self.w,test.transpose()))) < 0.5)


if __name__ == '__main__':
    bp = BP()
    data_path = '../data/train_binary.csv'
    frame = pd.read_csv(data_path)
    data = frame.values[:,1:]
    label = frame.values[:,0]
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size = 0.33,random_state = 23323)
    # print(train_data.shape,test_data.shape)
    bp.train(train_data,train_label)
    result = []
    for i, test in enumerate(test_data):
        result_i = bp.predict(test)
        # print(test_label[i] ,'compare --- ',result_i)
        result.append(result_i)
        if i % 10 == 0:
            accuracy = accuracy_score(test_label[:i + 1], result)
            print('accuracy is ', accuracy)
    print('accuracy is ', accuracy)
    # accuracy is  0.9831059129304743
