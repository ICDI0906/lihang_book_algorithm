# @Time : 2018/11/18 下午5:32 
# @Author : Kaishun Zhang 
# @File : compare.py 
# @Function: 比较自己和别人写的区别
import numpy as np

# np.argwhere(data) 同where 但是器其形式是array([[0, 0],[1, 1]]) 每一个元素表示一个点坐标
# np.where(data) # 索引列表,data必须为np.array()类型的数据，当data为多维的时候，array([0, 1]), array([0, 1]))其中表示两个点的坐标
# (0,0) 和 (1,1)
# np.index()可以得到最近的一个索引值最小的值
# 那份代码并没有考虑到当训练中的一个属性的所有值并没有在里面的时候，就是说训练数据中不包含所有的属性值
# 本来只有0.1 的正确率，发现是自己预测的时候写错了，改正后还是只有44% 的正确率，还是没有别人写的85%左右的正确率。
# 然后发现attributes 传值的使用应该使用局部变量，因为python函数传值只有党传递的是constant 才是局部变量，否则就是引用！！！
def fun(a):
    # print(a)
    if len(a) == 0:
        return
    a.remove(np.random.choice(a))
    for i in range(10):
        print(a)
        fun(a)


def a_add(a):
    # where a is constant，in the function it does not affect a, otherwise not
    for i,a_i in enumerate(a):
        a[i] +=1


if __name__ == '__main__':
    a = [1, 2, 3]
    a_add(a)
    print(a)