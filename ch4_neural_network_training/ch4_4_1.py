import numpy as np

from ch4_3_3 import function_2
from ch4_4 import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    # 学習によって自動的に獲得されるのではなく、
    # lr, step_numのような人の手によって設定されるパラメータを
    # ハイパーパラメータという
    print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
    print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
    print(gradient_descent(function_2, init_x=init_x, lr=1e-20, step_num=100))
