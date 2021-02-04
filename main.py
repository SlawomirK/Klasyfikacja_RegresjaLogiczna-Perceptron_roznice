import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from ClassyficationFactory import Perceptron, LogisticRegressionGD
from plotka import plot_decision_regions


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    # print('y = ', y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    y_train_03_subset = y_train.copy()
    y_train_01_subset = y_train.copy()
    x_train_01_subset = x_train.copy()

    clasyficationFactory = None
    clas = 'Perceptron'
   #clas = 'regLog'

    if clas == "Perceptron":
        clasyficationFactory = Perceptron()
        y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
        y_train_01_subset[(y_train_01_subset == 0)] = 1

        y_train_03_subset[(y_train == 1) | (y_train == 0)] = -1
        y_train_03_subset[(y_train_03_subset == 2)] = 1


    else:
        clasyficationFactory = LogisticRegressionGD()
        y_train_01_subset[(y_train == 1) | (y_train == 2)] = 1
        y_train_01_subset[(y_train_01_subset == 0)] = 0

        y_train_03_subset[(y_train == 1) | (y_train == 0)] = 1
        y_train_03_subset[(y_train_03_subset == 2)] = 0

    ppn1 = clasyficationFactory.startTrain(x_train_01_subset, y_train_01_subset)
    ppn3 = clasyficationFactory.startTrain(x_train_01_subset, y_train_03_subset)

    if clas == 'regLog':
        probabilityofLogicReggression(ppn1, ppn3, x_train_01_subset)

    classifier = clasyficationFactory.classifierFactory(ppn1, ppn3)

    plot_decision_regions(x_train, y_train, classifier=classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


def probabilityofLogicReggression(ppn1, ppn3, x_train_01_subset):
    print(ppn3.activation(ppn3.net_input(x_train_01_subset)))
    print(ppn1.activation(ppn1.net_input(x_train_01_subset)))


if __name__ == '__main__':
    main()
