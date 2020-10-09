import numpy as np

class Relu:
    """
    ReLUレイヤ
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # x: np.array型
        # self.mask: xと同じ形式の行列で、各要素0以下ならTrue/そうでなければfalse
        self.mask = (x <= 0)
        out = x.copy()
        # 各要素0以下で0になる
        out[self.mask] = 0

        return out

    def backword(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


if __name__ == '__main__':
    relu = Relu()
    x = np.array([[1, -2, 10], [-1, 10, 2]])
    out = relu.forward(x)
    print(f"out: {out}")
