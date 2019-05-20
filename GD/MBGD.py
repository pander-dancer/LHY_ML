import numpy as np
import random


class MBGD():
    def __init__(self, lr=0.001, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def fit(self, samples, y, batch_size):
        """
        MBGD 线性回归训练过程
        :param samples: 训练数据
        :param y: 预测值 groundtruth
        :param batch_size: mini_batch 占总体数据集的占比
        :return: 参数和每个epoch损失 w,cost
        """
        sample_num, dim = samples.shape
        y = y.flatten()
        w = np.ones((dim,), dtype=np.float)
        loss = 1
        iter_count = 0
        cost = []
        while loss > 0.0001 and iter_count < self.epochs:
            loss = 0
            error = np.zeros((dim,), dtype=np.float)

            # 随机采样后 算梯度
            index = random.sample(range(sample_num), int(np.ceil(sample_num * batch_size)))
            batch_samples = samples[index]
            batch_y = y[index]
            for i in range(len(batch_samples)):
                predict_y = np.dot(w.T, batch_samples[i])
                for j in range(dim):
                    error[j] += (predict_y - batch_y[i]) * batch_samples[i][j]
            # 更新参数
            for j in range(dim):
                w[j] -= self.lr * error[j] / sample_num

            # 算损失
            for i in range(sample_num):
                predict_y = np.dot(w.T, samples[i])
                error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
                loss += error
            cost.append(loss)
            # print('iter_count:', iter_count, 'loss:', loss)
            iter_count += 1
        return w, cost
