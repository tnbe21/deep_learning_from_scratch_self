import numpy as np

narr = np.array

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


if __name__ == '__main__':
    a = narr([0.3, 2.9, 4.0])
    exp_a = np.exp(a)
    print(exp_a)

    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)

    y = exp_a / sum_exp_a
    print(y)
