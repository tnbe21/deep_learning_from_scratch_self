import numpy as np


def perceptron(w1, w2, b, x1, x2):
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def AND(x1, x2):
    return perceptron(0.5, 0.5, -0.7, x1, x2)


def NAND(x1, x2):
    return perceptron(-0.5, -0.5, 0.7, x1, x2)


def OR(x1, x2):
    return perceptron(1.0, 1.0, -0.5, x1, x2)


def NOR(x1, x2):
    return perceptron(-0.5, -0.5, 0.4, x1, x2)
