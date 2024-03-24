import numpy as np


class SVM:
    def __init__(self, kernel, C, gamma, tol=0.1):
        # 初始化核函数
        self.kernel = kernel
        self.gamma = gamma
        # 初始化对偶问题参数
        self.alpha = None
        self.C = C
        # 初始化决策函数参数
        self.b = np.random.rand()
        self.x = None
        self.y = None
        self.tol = tol

    def fit(self, x, y, iterations=1000):
        """
        训练模型
        参数说明：   x n*features_n  np.array
                    y n-vector      np.array
        Using SMO algorithm
        """
        # SMO
        # 初始化
        self.x = x
        self.y = y
        self.alpha = np.random.rand(len(x)) * self.C
        alpha1 = None
        alpha2 = None
        alpha1_id = None
        alpha2_id = None
        E_all = np.array([self.E(x, y, x[i], y[i]) for i in range(len(y))])
        while iterations:
            # 选择第一个变量
            flag = False
            for i in range(len(x)):
                if self.C > self.alpha[i] > 0:
                    if abs(y[i] * self.g(x, y, x[i]) - 1) > self.tol:
                        alpha1 = self.alpha[i]
                        alpha1_id = i
                        flag = True
                        break
            if not flag:
                for i in range(len(x)):
                    if self.alpha[i] == 0:
                        if y[i] * self.g(x, y, x[i]) < 1 - self.tol:
                            alpha1 = self.alpha[i]
                            alpha1_id = i
                            flag = True
                            break
                    elif self.alpha[i] == self.C:
                        if y[i] * self.g(x, y, x[i]) > 1 + self.tol:
                            alpha1 = self.alpha[i]
                            alpha1_id = i
                            flag = True
                            break
            # 遍历完数据集后如果还没有发现违反KKT条件的样本点,则得到最优解
            if not flag:
                print("get optimal alpha")
                break
            # 选择第二个变量
            E_1 = E_all[alpha1_id]
            if E_1 >= 0:
                alpha2_id = np.argmin(E_all)
                alpha2 = self.alpha[alpha2_id]
            else:
                alpha2_id = np.argmax(E_all)
                alpha2 = self.alpha[alpha2_id]
            E_2 = E_all[alpha2_id]
            # 对alpha1 alpha2进行优化
            # 这里是解析解
            # 求alpha2的取值边界
            if y[alpha2_id] != y[alpha1_id]:
                L = np.max([0, alpha2 - alpha1])
                H = np.min([self.C, self.C + alpha2 - alpha1])
            else:
                L = np.max([0, alpha2 + alpha1 - self.C])
                H = np.min([self.C, alpha2 + alpha1])
            # eta = K11+K22-K12
            eta = (self.kernal_(x[alpha1_id], x[alpha1_id]) + self.kernal_(x[alpha2_id], x[alpha2_id]) - 2 *
                   self.kernal_(x[alpha1_id], x[alpha2_id]))
            alpha2_uncut = alpha2 + y[alpha2_id] * (E_1 - E_2) / eta
            if alpha2_uncut > H:
                alpha2 = H
            elif L <= alpha2_uncut <= H:
                alpha2 = alpha2_uncut
            else:
                alpha2 = L
            # 更新alpha
            alpha1_old = self.alpha[alpha1_id]
            alpha2_old = self.alpha[alpha2_id]
            self.alpha[alpha1_id] = alpha1 + y[alpha1_id] * y[alpha2_id] * (alpha2_old - alpha2)
            self.alpha[alpha2_id] = alpha2
            # 更新 b
            b1 = -E_1 - \
                 y[alpha1_id] * self.kernal_(x[alpha1_id], x[alpha1_id]) * (self.alpha[alpha1_id] - alpha1_old) - \
                 y[alpha2_id] * self.kernal_(x[alpha2_id], x[alpha1_id]) * (self.alpha[alpha2_id] - alpha2_old) + self.b
            b2 = -E_2 - \
                 y[alpha1_id] * self.kernal_(x[alpha1_id], x[alpha2_id]) * (self.alpha[alpha1_id] - alpha1_old) - \
                 y[alpha2_id] * self.kernal_(x[alpha2_id], x[alpha2_id]) * (self.alpha[alpha2_id] - alpha2_old) + self.b
            self.b = (b1 + b2) / 2
            # 更新E
            E_all = np.array([self.E(x, y, x[i], y[i]) for i in range(len(y))])
            iterations -= 1

    def predict(self, x):
        """
        预测函数
        输入:       x n*features_n np.array
        输出:       y_pre n-vector
        """
        y_pre = np.zeros(len(x))
        for i in range(len(x)):
            y_pre[i] = np.sum(self.alpha * self.y * self.kernal_(self.x, x[i]), axis=-1)
        y_pre += self.b
        y_pre = self.sign(y_pre)
        return y_pre

    def kernal_(self, x, z, p=2):
        gamma = self.gamma
        if self.kernel == 'linear':
            return x @ z
        elif self.kernel == 'poly':
            return np.power(x @ z + 1, p)  # 默认p=2
        elif self.kernel == 'rbf':
            return np.exp(-np.linalg.norm(x - z, axis=-1, ord=2) * gamma)
        else:
            raise Exception("核函数定义错误！！")

    def g(self, x, y, xi):
        return np.sum(self.alpha * y * self.kernal_(x, xi)) + self.b

    def E(self, x, y, xi, yi):
        return self.g(x, y, xi) - yi

    def sign(self, x):
        if type(x) == np.ndarray:
            x[x >= 0] = 1
            x[x < 0] = -1
        elif (type(x) == float) | (type(x) == int):
            if x >= 0:
                x = 1
            else:
                x = -1
        return x