import numpy as np

from ch3_5_1 import softmax as softmax_old

narr = np.array


def softmax_new(a):
    c = np.max(a)
    # 決まった数cを入れても変わらない
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

if __name__ == '__main__':
    a = narr([1010, 1000, 990])
    y_old = softmax_old(a)
    # オーバーフローする
    print(y_old)

    # オーバーフローしない
    y_new = softmax_new(a)
    print(y_new)
