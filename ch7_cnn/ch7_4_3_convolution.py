import os
import sys

import numpy as np

sys.path.append(os.pardir)

from common.util import im2col


x1 = np.random.rand(1, 3, 7, 7)
# x:
# [a1, a2, ..., a7,
# a8, a9, ..., a14,
# ...
# a43, a44, ..., a49] (ch1)
# [b1, b2, ..., b7,
# ...
# b43, b44, ..., b49] (ch2)
# [c1, c2, ..., c7,
# ...
# c43, c44, ..., c49] (ch3)
# shape: (1, 3, 7, 7) (batch size: 1)
#
# -> im2col(x):
# [a1, a2, ..., a5, a8, a9, ..., a12, ..., a33, b1, b2, ..., b33, c1,..., c33],
# [a2, a3, ..., a6, a9, a10, ..., a13, ..., a34, b2, b3, ..., b34, c2,..., c34],
# ...
# [a17, a18, ..., a21, a24, a25, ..., a28, ..., a49, b17, ..., b49, c17, ..., c49]

col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ(backward時に使用)
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # フィルタ(W)の
        # * FN: バッチ数
        # * C: チャンネル数
        # * FH: height
        # * FW: width
        FN, C, FH, FW = self.W.shape
        # 入力(x)の
        # * N: バッチ数
        # * C: チャンネル数
        # * H: height
        # * W: width
        N, C, H, W = x.shape
        out_h = int((H + 2 * self.pad - FH) / self.stride) + 1
        out_w = int((W + 2 * self.pad - FW) / self.stride) + 1

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        # transpose()を使い(N, C, H, W)にする
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
