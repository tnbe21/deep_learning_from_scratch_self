import numpy as np


def numerical_diff_before(f, x):
    """
    数値微分(cf: 解析的微分) 改善前
    """
    h = 10e-50
    # 前方差分
    return (f(x + h) - f(x)) / h


def numerical_diff(f, x):
    """
    数値微分 改善後
    """
    # 10e-50だと0.0に丸められてしまう
    h = 10e-4
    # 中心差分: 前方差分より誤差が減る
    return (f(x + h) - f(x - h)) / (2 * h)
