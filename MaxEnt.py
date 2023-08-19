import math
from collections import defaultdict
import numpy as np
from scipy.optimize import fminbound
import copy


class MaxEnt:
    def __init__(self, epsilon=1e-4, maxstep=200, algorithms='', verbose=False):
        print('this tool use a special way to get more eigenfunction')
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.w = None
        self.labels = None
        self.fea_list = []
        self.px = defaultdict(lambda: 0)
        self.pxy = defaultdict(lambda: 0)
        self.exp_fea = defaultdict(lambda: 0)
        self.data_list = []
        self.N = None
        self.M = None
        self.zw = None
        self.n_fea = None
        self.n_ilter = None
        self._algorithms = algorithms
        self._verbose = verbose

    def init_param(self, X_data, y_data):
        self.N = X_data.shape[0]
        self.labels = np.unique(y_data)

        self.fea_func(X_data, y_data)
        self.n_fea = len(self.fea_list)
        self.w = np.zeros(self.n_fea)
        self._exp_fea(X_data, y_data)
        return

    def fea_func(self, X_data, y_data, rules=None):

        if rules is None:
            for X, y in zip(X_data, y_data):
                X = tuple(X)
                self.px[X] += 1.0 / self.N
                self.pxy[(X, y)] += 1.0 / self.N
                for dimension, val in enumerate(X):
                    key = (dimension, val, y)
                    if not key in self.fea_list:
                        self.fea_list.append(key)
            self.M = X_data.shape[1]
        else:
            self.M = defaultdict(int)
            for i in range(self.N):
                self.M[i] = X_data.shape[1]
            pass

    def _exp_fea(self, X_data, y_data):

        for X, y in zip(X_data, y_data):
            for dimension, val in enumerate(X):
                fea = (dimension, val, y)
                self.exp_fea[fea] += self.pxy[(tuple(X), y)]
        return

    def _py_X(self, X):

        py_X = defaultdict(float)

        for y in self.labels:
            s = 0
            for dimension, val in enumerate(X):
                tmp_fea = (dimension, val, y)
                for i in self.fea_list:
                    if tmp_fea == i:
                        s += self.w[self.fea_list.index(tmp_fea)]
            py_X[y] = np.exp(s)

        normalizer = sum(py_X.values())
        self.zw = normalizer
        for key, val in py_X.items():
            py_X[key] = val / normalizer
        return py_X

    def _est_fea(self, X_data, y_data):

        est_fea = defaultdict(float)
        for X, y in zip(X_data, y_data):
            py_x = self._py_X(X)[y]
            for dimension, val in enumerate(X):
                est_fea[(dimension, val, y)] += self.px[tuple(X)] * py_x
        return est_fea

    def GIS(self, X_data, y_data):

        est_fea = self._est_fea(X_data, y_data)
        delta = np.zeros(self.n_fea)

        for j in range(self.n_fea):
            try:
                delta[j] = 1 / self.M * \
                    math.log(self.exp_fea[self.fea_list[j]
                                          ] / est_fea[self.fea_list[j]])
            except:
                continue
        delta = delta / delta.sum()

        return delta

    def IIS(self, delta, X_data, y_data):

        g = np.zeros(self.n_fea)
        g_diff = np.zeros(self.n_fea)
        for j in range(self.n_fea):
            for k in range(self.N):

                g[j] += self.px[tuple(X_data[k])] * self._py_X(X_data[k]
                                                               )[y_data[k]] * math.exp(delta[j] * self.M)
                g_diff[j] += self.px[tuple(X_data[k])] * self._py_X(
                    X_data[k])[y_data[k]] * math.exp(delta[j] * self.M) * self.M
            g[j] -= self.exp_fea[j]
            delta[j] -= g[j] / g_diff[j]
        return delta

    def fit(self, X_data, y_data):
        self.n_ilter = 0

        self.init_param(X_data, y_data)

        if self._algorithms == '':
            if isinstance(self.M, int):
                i = 0
                while i < self.maxstep:
                    self.n_ilter += 1
                    i += 1
                    delta = self.GIS(X_data, y_data)
                    if max(abs(delta)) < self.epsilon:
                        break
                    self.w += delta
            else:
                i = 0
                delta = np.random.rand(self.n_fea)
                while i < self.maxstep:
                    self.n_ilter += 1
                    i += 1
                    delta = self.IIS(delta, X_data, y_data)
                    if max(abs(delta)) < self.epsilon:
                        break
                    self.w += delta
        elif self._algorithms == 'IIS':
            if isinstance(self.M, int):
                print(
                    'Ur dataset\'s all features are same, so GIS algorithm will be used(In this case, IIS == GIS)')
                i = 0
                while i < self.maxstep:
                    self.n_ilter += 1
                    i += 1
                    delta = self.GIS(X_data, y_data)
                    if max(abs(delta)) < self.epsilon:
                        break
                    self.w += delta
            else:
                i = 0
                delta = np.random.rand(self.n_fea)
                while i < self.maxstep:
                    self.n_ilter += 1
                    i += 1
                    delta = self.IIS(delta, X_data, y_data)
                    if max(abs(delta)) < self.epsilon:
                        break
                    self.w += delta
        elif self._algorithms == 'GIS':
            if isinstance(self.M, int):
                i = 0
                while i < self.maxstep:
                    self.n_ilter += 1
                    i += 1
                    delta = self.GIS(X_data, y_data)
                    if max(abs(delta)) < self.epsilon:
                        break
                    self.w += delta
            else:
                print('Using IIS(different feature numbers among samples)')
                i = 0
                delta = np.random.rand(self.n_fea)
                while i < self.maxstep:
                    self.n_ilter += 1
                    i += 1
                    delta = self.IIS(delta, X_data, y_data)
                    if max(abs(delta)) < self.epsilon:
                        break
                    self.w += delta
        elif self._algorithms == 'DFP':

            print('Pleause the another module called MaxEntDFP')

        else:
            print('please decide algorithms in IIS&GIS&DFP, defult is GIS/IIS')
            return

        print('实际迭代次数{}'.format(self.n_ilter))

        return

    def predict(self, X):

        py_x = self._py_X(X)
        best_label = max(py_x, key=py_x.get)
        if self._verbose:
            print("模型权重：{}".format(self.w))
            print("模型特征函数量：{}".format(self.n_fea))
        return best_label


######################################################################################################################################
# DFP
######################################################################################################################################


class MaxEntDFP:
    def __init__(self, epsilon, max_iter=1000, distance=0.01):
        """
        最大熵的DFP算法
        :param epsilon: 迭代停止阈值
        :param max_iter: 最大迭代次数
        :param distance: 一维搜索的长度范围
        """
        print('this algorithms only support str data for now(23.08.19)')
        self.distance = distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.w = None
        self._dataset_X = None
        self._dataset_y = None

        self._y = set()
        self._xyID = {}
        self._IDxy = {}
        self._pxy_dic = defaultdict(int)
        self._N = 0
        self._n = 0
        self.n_iter_ = 0

    def init_params(self, X, y):
        self._dataset_X = copy.deepcopy(X)
        self._dataset_y = copy.deepcopy(y)
        self._N = X.shape[0]

        for i in range(self._N):
            xi, yi = X[i], y[i]
            self._y.add(yi)
            for _x in xi:
                self._pxy_dic[(_x, yi)] += 1

        self._n = len(self._pxy_dic)
        self.w = np.zeros(self._n)

        for i, xy in enumerate(self._pxy_dic):
            self._pxy_dic[xy] /= self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def calc_zw(self, X, w):
        zw = 0.0
        for y in self._y:
            zw += self.calc_ewf(X, y, w)
        return zw

    def calc_ewf(self, X, y, w):
        sum_wf = self.calc_wf(X, y, w)
        return np.exp(sum_wf)

    def calc_wf(self, X, y, w):
        sum_wf = 0.0
        for x in X:
            if (x, y) in self._pxy_dic:
                sum_wf += w[self._xyID[(x, y)]]
        return sum_wf

    def calc_pw_yx(self, X, y, w):
        return self.calc_ewf(X, y, w) / self.calc_zw(X, w)

    def calc_f(self, w):
        fw = 0.0
        for i in range(self._n):
            x, y = self._IDxy[i]
            for dataset_X in self._dataset_X:
                if x not in dataset_X:
                    continue
                fw += np.log(self.calc_zw(x, w)) - \
                    self._pxy_dic[(x, y)] * self.calc_wf(dataset_X, y, w)

        return fw

    # DFP
    def fit(self, X, y):
        self.init_params(X, y)

        def calc_dfw(i, w):

            def calc_ewp(i, w):
                ep = 0.0
                x, y = self._IDxy[i]
                for dataset_X in self._dataset_X:
                    if x not in dataset_X:
                        continue
                    ep += self.calc_pw_yx(dataset_X, y, w) / self._N
                return ep

            def calc_ep(i):
                (x, y) = self._IDxy[i]
                return self._pxy_dic[(x, y)]

            return calc_ewp(i, w) - calc_ep(i)

        def calc_gw(w):
            return np.array([[calc_dfw(i, w) for i in range(self._n)]]).T

        Gk = np.array(np.eye(len(self.w), dtype=float))

        w = self.w
        gk = calc_gw(w)
        if np.linalg.norm(gk, ord=2) < self.epsilon:
            self.w = w
            return

        k = 0
        for _ in range(self.max_iter):
            pk = -Gk.dot(gk)

            def _f(x):
                z = w + np.dot(x, pk).T[0]
                return self.calc_f(z)

            _lambda = fminbound(_f, -self.distance, self.distance)

            delta_k = _lambda * pk
            w += delta_k.T[0]

            gk1 = calc_gw(w)
            if np.linalg.norm(gk1, ord=2) < self.epsilon:
                self.w = w
                break
            yk = gk1 - gk
            Pk = delta_k.dot(delta_k.T) / (delta_k.T.dot(yk))
            Qk = Gk.dot(yk).dot(yk.T).dot(Gk) / (yk.T.dot(Gk).dot(yk)) * (-1)
            Gk = Gk + Pk + Qk
            gk = gk1

            k += 1

        self.w = w
        self.n_iter_ = k

    def predict(self, x):
        result = {}
        for y in self._y:
            prob = self.calc_pw_yx(x, y, self.w)
            result[y] = prob

        return result


# test function
def main(algorithms=''):
    if algorithms == 'DFP':
        dataset = np.array([['no', 'sunny', 'hot', 'high', 'FALSE'],
                            ['no', 'sunny', 'hot', 'high', 'TRUE'],
                            ['yes', 'overcast', 'hot', 'high', 'FALSE'],
                            ['yes', 'rainy', 'mild', 'high', 'FALSE'],
                            ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
                            ['no', 'rainy', 'cool', 'normal', 'TRUE'],
                            ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
                            ['no', 'sunny', 'mild', 'high', 'FALSE'],
                            ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
                            ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
                            ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
                            ['yes', 'overcast', 'mild', 'high', 'TRUE'],
                            ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
                            ['no', 'rainy', 'mild', 'high', 'TRUE']])

        X_train = dataset[:, 1:]
        y_train = dataset[:, 0]

        mae = MaxEntDFP(epsilon=1e-4, max_iter=1000, distance=0.01)
        mae.fit(X_train, y_train)
        print("模型训练迭代次数：{}次".format(mae.n_iter_))
        print("模型权重：{}".format(mae.w))

        result = mae.predict(['overcast', 'mild', 'high', 'FALSE'])
        print("预测结果：", result)
    else:
        print('running IIS/GIS')
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X_data = data['data']
        y_data = data['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.25, random_state=0)

        ME = MaxEnt(algorithms='')
        a = ME.fit(X_train, y_train)

        score = 0

        for X, y in zip(X_test, y_test):
            if ME.predict(X) == y:
                score += 1
        print(score / len(y_test) * 100)


if __name__ == '__main__':
    main('')
