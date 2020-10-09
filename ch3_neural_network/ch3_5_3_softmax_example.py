import numpy as np

from ch3_5_2_softmax import softmax_new as softmax

narr = np.array

if __name__ == '__main__':
    a = narr([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    # sumすると1
    print(np.sum(y))

    # softmaxは分類問題で用いるが、その出力は各aの確率と解釈できる
    # 機械学習の手順は学習と最終的な推論に分かれるが、
    # softmaxは指数関数で計算のリソースをそれなりに食うので、
    # 学習の際はsoftmaxを用いるが、最後の推論では使わないのが一般的
