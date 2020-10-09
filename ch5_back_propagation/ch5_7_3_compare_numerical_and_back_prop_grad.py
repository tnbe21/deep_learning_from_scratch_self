import os
import sys

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from ch5_7_2_two_layer_net_with_back_propagation import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 数値微分による勾配と誤差逆伝播法による勾配の比較
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f"{key}: {diff}")
