import os
import sys
sys.path.append(os.pardir)

import numpy as np

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        print(f"self.W: {self.W}")
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


if __name__ == '__main__':
    net = SimpleNet()
    print(net.W)

    # 入力
    ipt = np.array([0.6, 0.9])
    # 算出したラベル
    p = net.predict(ipt)
    print(p)

    print(np.argmax(p))

    # 正解ラベル
    t = np.array([0, 0, 1])
    # 損失関数の計算
    print(net.loss(ipt, t))


    def f(W):
        return net.loss(ipt, t)

    # net.Wの各要素についてのnet.loss(ipt, t)の偏微分結果が格納
    # (net.W自体をnumerical_gradientの中で変化させて計算(偏微分))
    dW = numerical_gradient(f, net.W)
    print(dW)