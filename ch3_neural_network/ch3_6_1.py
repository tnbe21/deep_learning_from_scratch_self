import sys, os
sys.path.append(os.pardir)
import numpy as np

from PIL import Image

from dataset.mnist import load_mnist

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__ == '__main__':
    # mnist.pyの使い方や学習、テストデータ形式の確認
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)

    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)

    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)
