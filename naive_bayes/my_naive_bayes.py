# @Time : 2018/11/16 下午2:02 
# @Author : Kaishun Zhang 
# @File : my_naive_bayes.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
import pandas as pd
import cv2

class Bayes(object):
    def __init__(self,lamda = 1): # lamda 拉普拉斯平滑
        self.p_c = dict() # p(c)
        self.p_cjl = dict()  # p(jl|c)
        self.lamda = lamda

    def binaryzation(self,img):
        cv_img = img.astype(np.uint8)
        # print(cv_img)
        _, cv_img = cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV)
        # print(cv_img.shape)
        # cv_img = img // 51 #取5个区间试一下
        return cv_img

    def train(self,train_data,train_label):
        size = train_data.shape[0]
        counter = Counter(train_label)
        for key,value in counter.items():
            self.p_c[key] = (value + self.lamda) / (size + self.lamda * len(counter))
        for key in counter.keys():
            sub_frame = train_data[train_data[:,0] == key]
            # 二值化
            sub_frame = self.binaryzation(sub_frame[:,1:])
            # print(sub_frame.shape)
            self.p_cjl[key] = dict()
            for j in range(0,sub_frame.shape[1]):
                counter = Counter(sub_frame[:,j])
                self.p_cjl[key][j] = dict()
                for key1,value1 in counter.items():
                    self.p_cjl[key][j][key1] = (value1 + self.lamda) / (sub_frame.shape[0] + len(counter) * self.lamda)

    def predict(self,test_data):
        result = dict()
        test_data = self.binaryzation(test_data[1:])
        for key in self.p_c.keys():
            result_tmp = self.p_c[key]
            for i,test in enumerate(test_data):
                if test[0] in self.p_cjl[key][i].keys():   # 对不在的直接pass
                    result_tmp *= self.p_cjl[key][i][test[0]]
            result[key] = result_tmp
        result = sorted(result.items(),key = lambda x:x[1],reverse = True)
        # print(result)
        return result[0][0]


if __name__ == '__main__':
    data_path = '../data/train.csv'
    frame = pd.read_csv(data_path)
    data = frame.values[:, 0:]
    label = frame.values[:, 0]
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size = 0.33,random_state = 23323)
    bayes = Bayes()
    bayes.train(train_data,train_label)
    result = []
    for i,test in enumerate(test_data):
        result_i = bayes.predict(test)
        # print(result_i ,' ---  ',test_label[i])
        result.append(result_i)
        if i % 100 ==0:
            accuracy = accuracy_score(test_label[:i + 1],result)
            print('accuracy is ',accuracy)
    accuracy = accuracy_score(test_label, result)
    print('accuracy is ', accuracy)
    # accuracy is  50 作为2值得话是 0.8277056277056277
    # accuracy is  120 作为2值得话是  0.8263347763347764