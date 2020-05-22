import numpy as np

from ch4_3_3 import function_2
from ch4_4 import _numerical_gradient_no_batch


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = _numerical_gradient_no_batch(f, x)
        x -= lr * grad

    return x


if __name__ == '__main__':
    # 学習によって自動的に獲得されるのではなく、
    # lr, step_numのような人の手によって設定されるパラメータを
    # ハイパーパラメータという
    print(gradient_descent(function_2, np.array([-3.0, 4.0]), 0.1, 100))
    print(gradient_descent(function_2, np.array([-3.0, 4.0]), 10.0, 100))
    print(gradient_descent(function_2, np.array([-3.0, 4.0]), 1e-10, 100))