import numpy as np
from torch import _test_serialization_subcmul
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import os

def load_dataset():
    train_dataset = h5py.File('D:/pho/Median/py/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('D:/pho/Median/py/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T



train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255




def sigmoid(z):
    """
    参数：
        z  - 任何大小的标量或numpy数组。

    返回：
        s  -  sigmoid（z）
    """
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    """
        此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。

        参数：
            dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）

        返回：
            w  - 维度为（dim，1）的初始化向量。
            b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros(shape = (dim,1))
    b = 0
    #使用断言来确保我要的数据是正确的
    assert(w.shape == (dim, 1)) #w的维度是(dim,1)
    assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者是int

    return (w , b)

def propagate(w, b, X, Y):
    """
    实现前向和后向传播的成本函数及其梯度。
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

    返回：
        cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """
    m = X.shape[1]

    #正向传播
    A = sigmoid(np.dot(w.T,X) + b) #计算激活值，请参考公式2。
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) #计算成本，请参考公式3和4。

    #反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T) #请参考视频中的偏导公式。
    db = (1 / m) * np.sum(A - Y) #请参考视频中的偏导公式。

    #使用断言确保我的数据是正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    #创建一个字典，把dw和db保存起来。
    grads = {
                "dw": dw,
                "db": db
             }
    return (grads , cost)

# 使用梯度下降法来优化w,b
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = True):
    """
    此函数通过运行梯度下降算法来优化w和b
    

    参数：
        w   - 权重，维度为(num_px * num_px * 3,1)的向量
        b   - 偏差
        X   - 维度为(num_px * num_px * 3,1)的向量
        Y   - 为真的“标签”向量(如果是猫为1，非猫为0),矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环迭代的次数
        learning_rate   - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值
    
    返回：
        params      - 包含权重w和偏差b的字典
        grads       - 包含权重和偏差相对于成本函数的梯度的字典
        costs        - 优化期间计算的所有成本列表，将用于绘制学习曲线

    """
    costs = []
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w -learning_rate * dw
        b = b - learning_rate * db

        #记录成本
        if i%100 == 0:
            costs.append(cost)
        if print_cost and i%100 == 0:
            #print ("Cost after iteration %i: %f" %(i, cost))
            print("迭代次数：%i ， 误差值： %f"% (i,cost))
    params = {
                "w":w,
                "b":b
    }
    grads = {
        "dw":dw,
        "db":db
    }
    return (params,grads,costs)

"""
print("*******testing for optimizing function*********")
w,b,X,Y = np.array([[1.],[2.]]),2.,np.array([[1.,2.,-1.],[3.,4.,-3.2]]),np.array([[1,0,1]])
params,grads,costs = optimize(w,b,X,Y,100,0.009,True)
print("w = "+str(params["w"]))
print("b = "+str(params["b"]))
print("dw = "+str(grads["dw"]))
print("db = "+ str(grads["db"]))

"""
# 实现预测函数predict()
def predict(w,b,X):
    """
    使用学习 逻辑回归参数logistic (w,b)预测标签是 0 还是 1 
    
    参数：
        w           - 权重，维度为(num_px * num_px * 3,1)的矩阵
        b           - 偏差
        X           - 维度为(num_px * num_px * 3,训练数据的数量)的矩阵

    返回:
        Y_prediction    - 包含X中所有图片的所有预测【0 | 1】的一个numpy 数组（向量)

    """
    m = X.shape[1] # 图片的数量
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    # 计算预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        #Y_prediction[0,i] = 1  if A[0,i] > 0.5 else 0
        Y_prediction[0,i] = (A[0,i]>0.5)

    assert(Y_prediction.shape == (1,m))
    return Y_prediction
'''
print("***************testing for predition function**********")

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print("预测结果： "+ str(predict(w,b,X)))
'''

# 将所有函数放入model()中

def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate = 0.5,print_cost = True):
    """
    调用之前实现的函数来构建逻辑回归模型

    参数：
        X_train     - 维度为(num_px * num_px * 3 ,m_train)的数组
        Y_train     - 维度为(1,m_train)的数组
        X_test      - 与上面类似
        Y_test      - 与上面类似
        num_iterations      - 用于优化参数的迭代次数的超参数
        learning_rate       - 表示optimize()更新规则中使用的学习速率的超参数
        print_cost          - 设置为True以100次迭代打印成本

    返回：
        d           - 包含有关模型信息的字典

    """

    w,b = initialize_with_zeros(X_train.shape[0])
    parameters ,grads , costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w,b = parameters["w"],parameters["b"]

    Y_prediction_test =predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    print("the accuracy of training datasets is : ",format(100-np.mean(np.abs(Y_prediction_train - Y_train))*100),"%")
    print("the accuracy of testing datasets is : ",format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100),"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")

    d = {
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations
    }

    return d

print("******************model function test************(******")
d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations = 2000,learning_rate = 0.01,print_cost = True)





costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Learning rate = ' +str(d["learning_rate"]))
plt.show()
