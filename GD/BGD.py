import numpy as np


class BGD():
    def __init__(self, lr=0.0001, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def fit(self, samples, y):
        sample_num, dim = samples.shape
        y = y.flatten()
        w = np.ones((dim,), dtype=np.float)
        loss = 10
        iter_count = 0
        cost = []
        while loss > 0.0001 and iter_count < self.epochs:
            loss = 0
            error = np.zeros((dim,), dtype=np.float)
            # 算梯度
            for i in range(sample_num):
                record = samples[i]
                predict_y = np.dot(w.T, samples[i])
                for j in range(dim):
                    error[j] += (-y[i] +predict_y) * samples[i][j]
            # 更新参数
            for j in range(dim):
                w[j] -= self.lr * error[j] / sample_num

            # 算loss
            for i in range(sample_num):
                predict_y = np.dot(w.T, samples[i])
                error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
                loss += error
            cost.append(loss)
            # print('iter_count:', iter_count, 'the loss:', loss)
            iter_count += 1
        return w, cost
