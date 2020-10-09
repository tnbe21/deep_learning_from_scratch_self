import numpy as np


def softmax(a):
    c = np.max(a)
    # 決まった数cを入れても変わらない
    exp_a = np.exp(a - c)
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


class SoftmaxWithLoss:
    """
    Softmax-with-Lossレイヤ
    """
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, t)

    def backward(self, dout=1):
        pass
