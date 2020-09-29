import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.pardir)
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 学習データを削減
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print(f'epoch:{str(epoch_cnt)}|{str(train_acc)}-{str(bn_train_acc)}')

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# グラフの描画
plot_nrows = 4
plot_ncols = 4
plot_num = plot_nrows * plot_ncols
# 初期値をログスケールで生成
weight_scale_list = np.logspace(0, -4, num=plot_num)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print(f'============== {i + 1} / {plot_num} ==============')

    # BNを用いた場合とそうでない場合両方での学習
    train_acc_list, bn_train_acc_list = __train(w)

    # グラフを1枚にweight_scale_listの要素数分表示
    plt.subplot(plot_nrows, plot_ncols, i + 1)

    plt.title(f'W: {str(w)}')

    # 最後の子グラフにだけ右下に凡例記載
    if i == plot_num - 1:
        # BNを用いた学習の精度(markevery: マーカーの表示頻度)
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        # BNを用いなかった学習の精度
        plt.plot(x, train_acc_list, linestyle='--', label='Normal(without BatchNorm)', markevery=2)
        plt.legend(loc='lower right')
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)

    plt.ylim(0, 1.0)

    # y軸ラベル設定
    if i % plot_ncols:
        plt.yticks([])
    else:
        plt.ylabel('accuracy')

    # x軸ラベル設定
    if i < plot_ncols * (plot_nrows - 1):
        plt.xticks([])
    else:
        plt.xlabel('epochs')

plt.show()
