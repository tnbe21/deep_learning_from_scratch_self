import os
import pickle
import sys
sys.path.append(os.pardir)

import numpy as np

from dataset.mnist import load_mnist

from my_common import stop_watch

from ch3_2_4 import sigmoid
from ch3_5_1 import softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)

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
def main():
    # 推論処理

    # テストデータ
    x, t = get_data()
    # サンプルパラメータ
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        # x[i]: 1つの画像(784(28*28)ピクセル)
        # t[i]: x[i]のラベル(正解の数字)

        # ラベリング出力
        y = predict(network, x[i])
        # 出力のうち確率最大のものを取得
        p = np.argmax(y)

        # テスト側ラベルと一致していれば正答数+1
        if p == t[i]:
            accuracy_cnt += 1

    print('Accuracy: ' + str(float(accuracy_cnt) / len(x)))

if __name__ == '__main__':
    main()
