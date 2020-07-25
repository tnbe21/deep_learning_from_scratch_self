import os
import sys

import numpy as np

sys.path.append(os.pardir)

from ch5_7_2 import TwoLayerNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# x_train -> t_train: 入力 -> 出力訓練データ
# x_train.shape: (入力数, 1入力次元数(784(28px*28px))
# t_train.shape: (出力数(=入力数), 1出力の次元数(10))
network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=50, output_size=t_train.shape[1])

# 試行回数
iters_num = 10000
# 訓練データ数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    # 全入出力の中からbatch_size分だけランダムにデータを抽出
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 誤差逆伝播法によって勾配を求める
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(train_acc, test_acc)
