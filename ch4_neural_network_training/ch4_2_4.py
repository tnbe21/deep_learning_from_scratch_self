import os
import pickle
import sys
sys.path.append(os.pardir)

import numpy as np

from my_common import stop_watch
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def cross_entropy_error(y, t):
    """
    交差エントロピー誤差出力
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # if文なしでこれでいける?
    # batch_size = y.shape[0] if y.ndim > 1 else 1
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def _cross_entropy_error(y, t):
    """
    交差エントロピー誤差出力
    書籍記載実装のif文使わないバージョン
    """
    batch_size = y.shape[0] if y.ndim > 1 else 1
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=True)

    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


@stop_watch
def exec_each():
    x, t = get_data()
    network = init_network()

    for i in range(len(x)):
        y = predict(network, x[i])
        err = cross_entropy_error(y, t[i])
        print(err)
        err = _cross_entropy_error(y, t[i])
        print(err)


@stop_watch
def exec_batch():
    x, t = get_data()
    network = init_network()

    batch_size = 10

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)
        err = cross_entropy_error(y_batch, t[i:i + batch_size])
        print(err)
        err = _cross_entropy_error(y_batch, t[i:i + batch_size])
        print(err)

if __name__ == '__main__':
    # 交差エントロピー誤差出力確認用スクリプト
    exec_each()
    exec_batch()
