import numpy as np
import matplotlib.pylab as plt

from ch4_3_1_numerical_diff import numerical_diff

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

if __name__ == '__main__':
    # np.ndarray type
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)

    diff_5 = numerical_diff(function_1, 5)
    diff_10 = numerical_diff(function_1, 10)
    print(diff_5)
    print(diff_10)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y)

    dy = numerical_diff(function_1, x)
    plt.plot(x, dy)

    # x = 5でのfunction_1の接線
    tangent_line_5 = diff_5 * x - 0.25
    plt.plot(x, tangent_line_5)

    # x = 10でのfunction_1の接線
    tangent_line_10 = diff_10 * x - 1
    plt.plot(x, tangent_line_10)

    plt.show()
