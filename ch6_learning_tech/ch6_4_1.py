import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.pardir)
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（荷重減衰）の設定
# 毎回の損失関数に対して1/2 * weight_decay_lambda * |W|^2を加算することで、一度での重みの補正を抑制し過学習を抑制する
# 1/2 * weight_decay_lambda * |W|^2: L2ノルム
# 1 * weight_decay_lambda * |W|: L1ノルム
# L2ノルムによる実行をL2正則化、L1ノルムだとL1正則化という

# 重み補正(訓練データによる訓練)で、典型的な訓練データから外れているものほど抑制度合いが強くなるイメージ
# 代わりに認識精度の向上も抑制される
# cf. オッカムの剃刀 「ある事柄を説明するためには必要以上に多くを仮定すべきではない」

# ※過学習が起きる主要因: パラメータを大量に持ち、表現力の高い(想定適用範囲が広すぎる)モデルであること, 訓練データが少ないこと

# weight decayを使用しない場合
# weight_decay_lambda = 0
weight_decay_lambda = 0.1

# weight_decay_lambda = 0(非正規化)と比べると、
# weight_decay_lambda非ゼロの場合(正規化されている場合)は
# 訓練データの認識精度とテストデータの認識精度との間の隔たりが小さくなる。代わりに両方で精度は全般的に減少

hidden_size_list = [100, 100, 100, 100, 100, 100]
network = MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配を求める
    grads = network.gradient(x_batch, t_batch)
    # 求めた勾配をもとにW, bを更新(学習)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f'epoch:{str(epoch_cnt)}, train acc:{str(train_acc)}, test acc:{str(test_acc)}')

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# network.params['W{n}']とnetwork.layers['Affine{n}'].Wの一致確認(動作確認用)
for layer_n in range(1, len(hidden_size_list) + 2):
    is_params_W_equal_to_layers_Affine_W = (network.params[f'W{layer_n}'] == network.layers[f'Affine{layer_n}'].W).all()
    print(f'is_params_W_equal_to_layers_Affine_W: {is_params_W_equal_to_layers_Affine_W}')

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
