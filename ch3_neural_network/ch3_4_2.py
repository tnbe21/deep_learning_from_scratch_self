import numpy as np

from ch3_2_4 import sigmoid

def identity_function(x):
    return x

if __name__ == '__main__':
    narr = np.array
    X = narr([1.0, 0.5])
    W1 = narr([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = narr([0.1, 0.2, 0.3])

    print(W1.shape)
    print(X.shape)
    print(B1.shape)

    # 0->1層入力信号総和
    A1 = np.dot(X, W1) + B1
    # sigmoid=活性化関数、Z1=1層の出力, 2層への入力
    Z1 = sigmoid(A1)

    W2 = narr([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = narr([0.1, 0.2])

    # 1->2層入力信号総和
    A2 = np.dot(Z1, W2) + B2
    # sigmoid=活性化関数、Z2=2層の出力, 出力層への入力
    Z2 = sigmoid(A2)

    print(Z2)
    W3 = narr([[0.1, 0.3], [0.2, 0.4]])
    B3 = narr([0.1, 0.2])

    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)
