# @Time : 2018/11/18 下午1:46
# @Author : Kaishun Zhang
# @File : my_ID3实现.py
# @Function:使用ID3算法实现对mnist数据集进行分类
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from queue import Queue
import time
fw = open('result_me','w')

class Node(object):

    def __init__(self,is_leaf,label,attribute):
        self.is_leaf = is_leaf
        self.label = label
        self.attribute = attribute
        self.children = dict()  #直接

class ID3(object):
    '''
        算法流程 李航p63
        输入：雪莲数据集D，特征集A，阈值epsilon
        (1) 若D中所有的实例属于同一类Ck,则将类Ck作为该节点的类标记
        (2) 若A=None,则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记。返回T
        (3) 否则，计算A中的各个特征对D的信息增益，选择信息增益最大的特征Ag
        (4) 如果Ag的信息增益小于阈值epsilon。则置T为单节点树，并将D中的实例树最大的类Ck,作为该节点的类标记，返回T
        (5) 否则对Ag的每一个可能值ai,依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子节点，由结点及其子节点构成树T，返回T
        (6) 对于第i个子节点，以Di为训练集，以A-{Ag}为特征集，递归调用(1) - (5),得到子树Ti,返回Ti
    '''
    def __init__(self,epsilon):
        self.epsilon = epsilon

    def entropy(self,x):
        total = sum(x)
        return -sum(x / total * np.log2(x / total))

    def binaryzation(self,img):
        cv_img = img.astype(np.uint8)
        _, cv_img = cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV)
        return cv_img

    def train(self,data,label,attributes):
        self.attribute = attributes  # 属性列
        self.attribute_value = dict()
        self.labels = list(set(label))
        for attribute_i in self.attribute:
            self.attribute_value[attribute_i] = tuple(Counter(data[:,attribute_i]).keys()) # every attribute have and cannot change
        self.root = self._train(data,label,attributes)

    def _train(self,data,label,attributes):
        # compute
        counter = Counter(label)
        mx = -1
        label_selected = -1
        for key, value in counter.items():
            if value > mx:
                mx = value
                label_selected = key
        if label_selected == -1:
            print('there is something wrong!! 1')

        if len(counter) == 1:
            # fw.write('1' + ' : '+str(label[0]) + '\n')
            node = Node(True,label[0],None)
            return node
        if len(attributes) == 0:
            # fw.write('2'+ ' : '+ str(label_selected) + '\n')
            node = Node(True,label_selected,None)
            return node
        entropy_count = np.array(list(counter.values()))
        entropy_data = self.entropy(entropy_count)
        # print('entropy_data',entropy_data)
        mx_entropy = -1e9   #最大的信息增益
        slice_attribute = -1 # 切分的属性
        for attribute_i in attributes: # 计算每一个属性的熵
            sum = 0
            # for attribute_value_i in self.attribute_value[attribute_i]: # 对于某个属性，某个类别
            for attribute_value_i in Counter(data[:,attribute_i]):
                index = data[:,attribute_i] == attribute_value_i
                data_i = data[index]
                label_i = label[index]
                if len(data_i) == 0:
                    continue
                sum_tmp_1 = len(data_i) / len(data)
                sum_tmp_2 = 0
                # for label_j in self.labels:
                for label_j in set(label_i):
                    data_j = data_i[label_i == label_j]
                    # print(data_j)
                    if len(data_j) == 0:# without this,nan occur
                        continue
                    len_1 = len(data_j)
                    len_2 = len(data_i)
                    sum_tmp_2 += (len_1 / len_2 * np.log2(len_1 / len_2))
                sum += (sum_tmp_1 * sum_tmp_2)
            entropy_tmp = entropy_data + sum
            # print('attributes_i --- ',attribute_i, '信息增益 -- ',entropy_tmp)
            if attribute_i == 130:
                fw.write(str(mx_entropy) + '\n')
            if mx_entropy < entropy_tmp:
                mx_entropy = entropy_tmp
                slice_attribute = attribute_i
        if mx_entropy < self.epsilon: # 小于最小的信息增益
            # fw.write('3'+' : ' + str(label_selected) + '\n')
            node = Node(True, label_selected,None)
            return node

        # print('slice_attribute ------ ', slice_attribute, 'label_selected ----- ', mx_entropy)
        fw.write(str(mx_entropy) + ' ' + str(slice_attribute)+ '\n')
        node = Node(False,label_selected,slice_attribute)
        sub_attribute = []
        for attribute_i in attributes:
            if attribute_i != slice_attribute:
                sub_attribute.append(attribute_i)      # important 应该使用局部变量，不要使用attributes.remove(attribute_i) 这是个大坑
        # for attribute_value_i in self.attribute_value[slice_attribute]:
        for attribute_value_i in Counter(data[:,slice_attribute]).keys():
            index = data[:,slice_attribute] == attribute_value_i
            sub_data = data[index]
            sub_label = label[index]
            # node.edge[attribute_value_i] = len(node.edge)
            if len(sub_data) == 0: # 使用父节点的类别做为自身类别
                node.children[attribute_value_i] = (Node(True,node.label,None))
            else:
                node.children[attribute_value_i] = self._train(sub_data,sub_label,sub_attribute)
        return node

    def bfs(self):
        root = self.root
        self._bfs(root)

    def _bfs(self,root):
        que = Queue()
        que.put(root)
        while not que.empty():
            q = que.get()
            if q.is_leaf:
                print('label --- ',q.label)
            print('slice ',q.attribute)
            for edge in q.edge.keys():
                print('start edge ---  ', edge ,'\n')
                que.put(q.children[q.edge[edge]])
                print('end edge ---  ', '\n')
    def dfs(self,test):
        root = self.root
        return self._dfs(root,test)

    def _dfs(self,root,test):
        if root.is_leaf:
            return root.label
        return self._dfs(root.children[test[root.attribute]],test) #important


if __name__ == '__main__':
    id3 = ID3(0.1)
    data_path = '../data/train.csv'
    # data_path = 'watermelon3.csv'
    frame = pd.read_csv(data_path)
    data = frame.values[:,1:]
    label = frame.values[:,0]
    data = id3.binaryzation(data)
    # print(data.shape,label.shape)
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size = 0.33,random_state = 23323)
    # print(train_data[:1])
    print('pre --------------------- ')
    start = time.time()
    id3.train(train_data,train_label,list(range(0,train_data.shape[1])))
    end = time.time()
    print('train cost {0} s'.format(end - start))
    result = []
    start = time.time()
    for i,test in enumerate(test_data):
        result_i = id3.dfs(test)
        # print(test_label[i] ,'compare --- ',result_i)
        result.append(result_i)
        if i % 10 == 0:
            accuracy = accuracy_score(test_label[:i + 1], result)
            print('accuracy is ', accuracy)
    end = time.time()

    print('test cost {0} s'.format(end - start))
    accuracy = accuracy_score(test_label, result)
    print('accuracy is ', accuracy)
    # accuracy is 0.8598845598845599