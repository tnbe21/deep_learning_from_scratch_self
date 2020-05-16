import numpy as np

from ch3_2_4 import sigmoid
from ch3_4_2 import identity_function

narr = np.array

def init_network():
    network = {}

    network['W1'] = narr([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = narr([0.1, 0.2, 0.3])
    network['W2'] = narr([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = narr([0.1, 0.2])
    network['W3'] = narr([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = narr([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    # 回帰問題(数値を予測する問題)では活性化関数はこの恒等関数でOK
    # a3の時点でパラメータ推定を終わらせているようなもので、
    # 出力はそのままa3のものを出す(という理解で合ってる?)
    y = identity_function(a3)

    return y


if __name__ == '__main__':
    network = init_network()
    x = narr([1.0, 0.5])
    y = forward(network, x)
    print(y)
