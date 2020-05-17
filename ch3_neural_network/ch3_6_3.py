import os
import sys
sys.path.append(os.pardir)

import numpy as np

from my_common import stop_watch

from ch3_6_2 import get_data, init_network, predict

@stop_watch
def main():
    # 3_6_2では1枚ずつ(x[i]ずつ)処理してたがここではまとめて処理してみる(バッチ)
    # 3_6_2より実行時間が短い
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i + batch_size])

    print('Accuracy: ' + str(float(accuracy_cnt) / len(x)))

if __name__ == '__main__':
    main()
