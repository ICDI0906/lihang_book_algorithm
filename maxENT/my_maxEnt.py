# @Time : 2018/11/20 下午6:21 
# @Author : Kaishun Zhang 
# @File : my_maxEnt.py 
# @Function:
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import time
from collections import defaultdict
class maxEnt(object):
    '''
    我一直很困惑，这个特征函数应该怎么来定义，借鉴于WenDesi的构建特征的方法
    使用梯度下降的方法尝试一下，这里梯度下降他取了一个log让正确率提高了一半，我觉得是因为梯度太小的原因！！
    但是觉得WenDesi 前后有点不一致，因为他构造特征函数的时候是把每一个拆开来的。形式为(id_x,y)
    所以这样的话，这样的话每一条数据就会多出很多的特征。(但是这正是我们想要的，这样子的话，特征函数的期望就
    有可能不为1，如果将一个数据的特征组合在一起形成一个特征函数的话，会使得泛化能力变得很差。而特征函数相当于一种规则)
    我原来的想法是既然分开，就尝试按照每一个一条记录这样子来算。但是又在想pyx应该怎么来算。
    原来想着是取多数。但这这样子会出现前后矛盾的情况。而且不是特别好算。更新的参数的时候可以想成是数据
    对fi(x,y)的贡献。

    对参数w的更新可以是梯度下降，也可以是改进的迭代尺度算法 或者 牛顿法，拟牛顿法

    有不同就会有思考，有思考就会有收获
    '''
    def __init__(self,max_iterator = 10,learning_rate = 1e-5):
        self.max_iterator = max_iterator
        self.learning_rate = learning_rate
    def train(self,train_data,train_label):
        self.X = train_data
        self.Y = set(train_label)

        self.build_feature_function(train_label)
        self.n = len(self.ff2int) # 选取的特征的个数
        self.calc_Epxy()

        # self.calc_Epx() # (p(y|x)的初始值 = 1/len(Y))
        self.w = np.zeros(self.n) # 初始化权重
        for iterator_i in range(self.max_iterator):
            print('in {} step'.format(iterator_i))
            self.calc_Epx()
            for i in range(self.n):
                self.w[i] += self.learning_rate * np.log(self.Epxy[i] / self.Epx[i]) # 算是处理小技巧吧，加上log

    def calc_pyx_numberator(self,X,y):
        result = 0.0
        for x in X:
            if (x,y) in self.ff2id.keys():
                result += self.w[self.ff2id[(x,y)]]
        return (np.exp(result),y)


    def calc_pro(self,X):
        result = [self.calc_pyx_numberator(X,y) for y in self.Y]
        zx = sum([pro for (pro,y) in result])
        return [(pro / zx,y) for pro,y in result]


    def calc_Epx(self):
        self.Epx = np.zeros(self.n)
        for data_i in self.X:
            pyxs = self.calc_pro(data_i)
            # 这里算的也是所有的data_i，下面又迭代了所有的data_i,难道不会重复的吗？
            # 或者说，是把所有的结果都算出来，看看那个(x,y)对Epx(i)有贡献，应该是这样的。
            for x in data_i:
                for (pro,y) in pyxs:
                    if (x,y) in self.ff2id.keys():
                        self.Epx[self.ff2id[(x,y)]] += pro * 1 / len(self.X)


    def calc_Epxy(self):
        '''
            计算特征函数f(x,y)关于经验分布p(x,y)的期望值。
            这里使用的特征是f(x,y) = 1,所以这里就是概率
        '''
        self.Epxy = dict()
        for key,value in self.ff2int.items():
            self.Epxy[self.ff2id[key]] = value / len(self.X) # 某一个id对应特征 对应的概率。

    def build_feature_function(self,train_label):
        self.id2ff = dict()#
        self.ff2id = dict()
        self.ff2int = defaultdict(int)
        self.x2int = defaultdict(int)
        for train_data_i,train_label_i in zip(self.X,train_label):
            for train_data_i_j in train_data_i:
                self.x2int[train_data_i_j] += 1
                if not (train_data_i_j,train_label_i) in self.ff2id.keys():
                    size = len(self.ff2id)
                    self.ff2id[(train_data_i_j,train_label_i)] = size
                    self.id2ff[size] = (train_data_i_j,train_label_i)
                    self.ff2int[(train_data_i_j,train_label_i)] = 1
                else:
                    self.ff2int[(train_data_i_j,train_label_i)] += 1

    def predict(self,test):
        result = self.calc_pro(test)
        result = sorted(result,key =lambda x:x[0],reverse = True)
        return result[0][1]


def rebuildFeature(data):

    result = []
    for data_i in data:
        tmp = []
        for i,data_i_j in enumerate(data_i):
            tmp.append(str(i) + '_' + str(data_i_j))
        result.append(tmp)
    return np.array(result)


if __name__ == '__main__':
    maxent = maxEnt()
    frame = pd.read_csv('../data/mini_train_binary.csv')
    data = frame.values[:,1:]
    label = frame.values[:,0]
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size = 0.33,random_state = 23323)
    train_data = rebuildFeature(train_data)
    test_data = rebuildFeature(test_data)
    maxent.train(train_data,train_label)
    result = []
    start = time.time()
    for i, test in enumerate(test_data):
        result_i = maxent.predict(test)
        print(test_label[i] ,'compare --- ',result_i)
        result.append(result_i)
        if i % 10 == 0:
            accuracy = accuracy_score(test_label[:i + 1], result)
            print('accuracy is ', accuracy)
    end = time.time()
    print(maxent.w)
    print('test cost {0} s'.format(end - start))
    accuracy = accuracy_score(test_label, result)
    print('accuracy is ', accuracy)