"""
    预测pm2.5
    这里我们用前九个小时pm2.5来预测第10小时的pm2.5
    基于AdaGrad
"""
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GD.AdaGrad import AdaGrad
from GD.BGD import BGD
from GD.SGD import SGD
from GD.MBGD import MBGD

"""
从训练数据中提取出连续十个小时的观测数据，
最后一个小时的PM2.5作为该条数据的类标签，
而前九个小时的PM2.5值作为特征。
一天24个小时，一天内总共有24-10+1 =15条记录
"""


def get_train_data():
    data = pd.read_csv("train.csv")
    # 获取所有的pm2.5信息
    pm2_5 = data[data['observation'] == 'PM2.5'].iloc[:, 3:]
    xlist = []
    ylist = []
    for i in range(15):
        tempx = pm2_5.iloc[:, i:i + 9]  # 使用前9小时数据作为feature
        tempx.columns = np.array(range(9))
        tempy = pm2_5.iloc[:, i + 9]  # 使用第10个小数数据作为lable
        tempy.columns = [1]
        xlist.append(tempx)
        ylist.append(tempy)
    xdata = pd.concat(xlist)
    ydata = pd.concat(ylist)
    X = np.array(xdata, float)
    # 加上bias
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = np.array(ydata, float)
    return X, y


def get_test_data():
    data = pd.read_csv("test.csv")
    pm2_5 = data[data['AMB_TEMP'] == 'PM2.5'].iloc[:, 2:]
    X = np.array(pm2_5, float)
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X


def Adagrad_f(X_train, X_test, y_train):
    gd = AdaGrad()
    w, cost = gd.fit(X_train, y_train)
    # print(w, cost[-1])

    # 预测
    h = np.dot(X_test, w)
    real = pd.read_csv('answer.csv')
    erro = abs(h - real.value).sum() / len(real.value)
    print('平均绝对值误差', erro)

    plt.title("AdaGrad linear regression")
    plt.plot(cost)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Result\\AdaGrad.png')
    # plt.show()


def BGD_f(X_train, X_test, y_train):
    bgd = BGD()
    w, cost = bgd.fit(X_train, y_train)
    # print(w, cost)
    # 预测
    h = np.dot(X_test, w)
    real = pd.read_csv('answer.csv')
    erro = abs(h - real.value).sum() / len(real.value)
    print('平均绝对值误差', erro)

    plt.title("BGD linear regression")
    plt.plot(cost)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Result\\BGD.png')
    # plt.show()


def SGD_f(X_train, X_test, y_train):
    sgd = SGD()
    w, cost = sgd.fit(X_train, y_train)
    # print(w,cost)
    # 预测
    h = np.dot(X_test, w)
    real = pd.read_csv('answer.csv')
    erro = abs(h - real.value).sum() / len(real.value)
    print('平均绝对值误差', erro)

    plt.title("SGD linear regression")
    plt.plot(cost)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Result\\SGD.png')
    # plt.show()


def MBGD_f(X_train, X_test, y_train):
    mbgd = MBGD()
    w, cost = mbgd.fit(X_train, y_train, 0.3)
    # 预测
    h = np.dot(X_test, w)
    real = pd.read_csv('answer.csv')
    erro = abs(h - real.value).sum() / len(real.value)
    print('平均绝对值误差', erro)

    plt.title("MBGD linear regression")
    plt.plot(cost)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Result\\MBGD.png')
    # plt.show()


if __name__ == '__main__':
    X_train, y_train = get_train_data()
    X_test = get_test_data()

    start = time.time()
    BGD_f(X_train, X_test, y_train)  # 平均绝对值误差 5.320293957865748
    print('BGD time:', time.time() - start)  # BGD time: 38.274808168411255

    start = time.time()
    SGD_f(X_train, X_test, y_train)  # 平均绝对值误差 5.007611937098004
    print('SGD time:', time.time() - start)  # SGD time: 59.29731798171997

    start = time.time()
    MBGD_f(X_train, X_test, y_train)  # 平均绝对值误差 5.163666086828864
    print('MBGD time:', time.time() - start)  # MBGD time: 20.305829286575317

    """
    Adagrad 对学习率约束 前期正则项较大 逐渐减小；仍然依赖人工设置的全局学习率，过大也不行
    """
    start = time.time()
    Adagrad_f(X_train, X_test, y_train)  # 平均绝对值误差 4.97442948413227
    print('Adagrad time:', time.time() - start)  # Adagrad time: 0.7410688400268555

    """
    跑的时候发现：
        学习率过大会有两个问题：① 调节幅度过大，导致loss产生较大波动；② 梯度弥散；
    """

