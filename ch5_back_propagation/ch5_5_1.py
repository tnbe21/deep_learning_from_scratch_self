import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # np.array型のxが入力される想定で、
        # self.maskはnp.array型xの各要素が0以下かどうかの論理値のnp.array
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backword(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
