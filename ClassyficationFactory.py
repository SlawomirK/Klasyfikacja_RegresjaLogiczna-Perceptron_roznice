from abc import ABC

import numpy as np


class ClassyficationFactory(ABC):

    # abstract method
    def fit(self, X, y):
        pass

    def net_input(self, X):
        pass

    def predcit(self, X):
        pass

    def classifierFactory(self, ppn1, ppn3):
        pass

    def startTrain(x_train_subset, y_train_subset):
        pass


class ClassifierLog(object):
    def __init__(self, ppn1, ppn3):
        self.ppn1 = ppn1
        self.ppn3 = ppn3

    def predict(self, x):
        return np.where(self.ppn1.predict(x) == 0, 0, np.where(self.ppn3.predict(x) == 0, 2, 1))


class LogisticRegressionGD(ClassyficationFactory):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def startTrain(self, x_train_subset, y_train_subset):
        p = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
        p.fit(x_train_subset, y_train_subset)
        return p

    def classifierFactory(self, ppn1, ppn3):
        return ClassifierLog(ppn1, ppn3)


class ClassifierP(object):
    def __init__(self, ppn1, ppn3):
        self.ppn1 = ppn1
        self.ppn3 = ppn3

    def predict(self, x):
        return np.where(self.ppn1.predict(x) == 1, 0, np.where(self.ppn3.predict(x) == 1, 2, 1))


class Perceptron(ClassyficationFactory):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def classifierFactory(self, ppn1, ppn3):
        return ClassifierP(ppn1, ppn3)

    def startTrain(self, x_train_subset, y_train_subset):
        p = Perceptron(eta=0.1, n_iter=50)
        p.fit(x_train_subset, y_train_subset)
        return p
