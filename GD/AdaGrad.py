import numpy as np


class AdaGrad():
    def __init__(self, lr=10, epochs=10000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, x, y):
        """
        """
        m = y.size
        w = np.zeros(x.shape[1])
        s_grad = np.zeros(x.shape[1])
        cost = []
        for j in range(self.epochs):
            y_hat = np.dot(x, w)                        # 模型函数_线性回归 y = w*x
            error = y_hat - y                           # error = h(x)-y
            grad = np.dot(x.transpose(), error)         # grad = (h(x)-y)x
            s_grad += grad ** 2                         # 梯度的平方和
            ada = np.sqrt(s_grad)                       #
            w -= self.lr * grad / ada                   # 参数更新
            J = 1.0 / (2) * (np.sum(np.square(y_hat - y))) # 损失计算
            cost.append(J)

        return w, cost
