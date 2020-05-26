import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


if __name__ == '__main__':
    sigmoid = Sigmoid()
    out = sigmoid.forward(np.array([[1.0, 5.0, -2.0], [-1.0, 3.0, 4.0]]))
    bk = sigmoid.backward(np.array([[2.0, 5.0, -2.0], [-1.0, 2.0, 4.0]]))

    print(f"bk: {bk}")
    print(f"out: {out}")
