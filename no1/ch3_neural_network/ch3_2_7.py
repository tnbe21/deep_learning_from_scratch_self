import numpy as np
import matplotlib.pylab as plt

def reLU(x):
    return np.maximum(0, x)

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = reLU(x)
    plt.plot(x, y)
    plt.ylim(-1.0, 5.5)
    plt.show()

