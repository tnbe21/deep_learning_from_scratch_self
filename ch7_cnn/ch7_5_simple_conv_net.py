import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle

from collections import OrderedDict

sys.path.append(os.pardir)
from common.gradient import numerical_gradient
from common.layers import *
from common.trainer import Trainer
from dataset.mnist import load_mnist


class SimpleConvNet:
    """単純なConvNet

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_dim : tuple of int
        入力データの次元((チャンネル, 高さ, 幅))
    conv_param : dictionary
        畳み込み層のハイパーパラメータ
        filter_num : int
            フィルタ数
        filter_size : int
            フィルタサイズ
        pad : int
            パディング
        stride : int
            ストライド
    hidden_size : int
        隠れ層(全結合層)のニューロンの数
    output_size : int
        出力層(全結合層)のニューロンの数
    weight_init_std : float
        初期化時の重みの標準偏差
    loaded_params_file_name : string
        重みとバイアスの読み込みファイル名
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'stride': 1, 'pad': 0},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_stride = conv_param['stride']
        filter_pad = conv_param['pad']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """推論を行う

        Parameters
        ----------
        x : numpy.ndarray
            入力データ

        Returns
        -------
        numpy.ndarray
            最後の出力層直前の出力
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数の値を求める

        Parameters
        ----------
        x : numpy.ndarray
            入力データ
        t : numpy.ndarray
            教師ラベル
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """認識精度を求める

        Parameters
        ----------
        x : numpy.ndarray
            入力データ
        t : numpy.ndarray
            教師ラベル
        batch_size : int
            バッチサイズ(デフォルト値は100)

        Returns
        -------
        float
            認識率
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """勾配を求める(数値微分)

        Parameters
        ----------
        x : numpy.ndarray
            入力データ
        t : numpy.ndarray
            教師ラベル

        Returns
        -------
        dictionary
            各層の勾配を持ったディクショナリ変数
            W{n}は各層の重み
            b{n}は各層のバイアス
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads[f'W{idx}'] = numerical_gradient(loss_w, self.params[f'W{idx}'])
            grads[f'b{idx}'] = numerical_gradient(loss_w, self.params[f'b{idx}'])

        return grads

    def gradient(self, x, t):
        """勾配を求める(誤差逆伝搬法)

        Parameters
        ----------
        x : numpy.ndarray
            入力データ
        t : numpy.ndarray
            教師ラベル

        Returns
        -------
        dictionary
            各層の勾配を持ったディクショナリ変数
            W{n}は各層の重み
            b{n}は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        for layer in reversed(layers):
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def save_params(self, file_name='params.pkl'):
        """重みとバイアスを保存する

        Parameters
        ----------
        file_name : string
            保存ファイル名
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        """重みとバイアスを読み込む

        Parameters
        ----------
        file_name : string
            読み込みファイル名
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params[f'W{i}']
            self.layers[key].b = self.params[f'b{i}']

if __name__ == '__main__':
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

    # 処理に時間のかかる場合はデータを削減
    # x_train, t_train = x_train[:5000], t_train[:5000]
    # x_test, t_test = x_test[:1000], t_test[:1000]

    max_epochs = 20

    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # パラメータの保存
    network.save_params()
    print('Saved Network Parameters!')

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
