# @Time : 2018/11/14 下午7:15 
# @Author : Kaishun Zhang 
# @File : my_perceptron.py 
# @Function: 小实验生成感知机模型
# 实验过程:
# 先生成一些x属于[0,1) y属于[0,1)的数据
# 然后用y = -x + 1对数据进行标记
# 然后通过SGD进行参数更新
# 最后将结果进行可视化
import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self,iterator_num = 100 ,learning_rate = 1e-4):
        # y = -x + 1
        self.iterator_num = iterator_num
        self.learning_rate = learning_rate
        x = np.random.rand(20)
        y = np.random.rand(20)
        self.data = []
        self.label = []
        for i,j in zip(x,y):
            self.data.append([i,j])
            if i + j > 1:
                self.label.append(1)
            else:
                self.label.append(-1)
        self.data = np.array(self.data)
        self.label = np.array(self.label)

    def plot_data(self):
        plt.scatter(self.data[:,0],self.data[:,1])
        plt.show()

    def train(self):
        self.w = np.zeros(self.data.shape[1])
        self.b = 0
        for iterator_i in range(self.iterator_num):
            for i,xi in enumerate(self.data):
                if self.label[i] * (np.dot(self.w,np.transpose(xi)) + self.b) <= 0: # mistake point
                    self.w += self.learning_rate * self.label[i] * xi
                    self.b += self.learning_rate * self.label[i]
        x = np.linspace(0,1,20)
        y = -(self.w[0] * x + self.b) / self.w[1]
        plt.plot(x,y)
        plt.scatter(self.data[:,0],self.data[:,1],c = self.label)
        # plt.show()
        plt.savefig('perceptron_result.jpg')


if __name__ == '__main__':
    perceptron = Perceptron(iterator_num = 10000)
    perceptron.train()