import matplotlib.pylab as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from ch4_3_1_numerical_diff import numerical_diff


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def function_tmp0(x_0):
    return function_2(np.array([x_0, 4.0]))


def function_tmp1(x_1):
    return function_2(np.array([3.0, x_1]))


if __name__ == '__main__':
    # x_0 = 3, x_1 = 4のときのx_0に対する偏微分
    print(numerical_diff(function_tmp0, 3.0))

    # x_0 = 3, x_1 = 4のときのx_1に対する偏微分
    print(numerical_diff(function_tmp1, 4.0))

    x_0 = x_1 = np.arange(-3.0, 3.0, 0.1)
    X_0, X_1 = np.meshgrid(x_0, x_1)
    Y = function_2(np.array([X_0, X_1]))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x_0")
    ax.set_ylabel("x_1")
    ax.set_zlabel("f(x)")

    ax.plot_wireframe(X_0, X_1, Y)
    plt.show()
