import numpy as np

def mapY(y, num_labels):
    # 样本数
    m = len(y);

    E = np.mat(np.eye(num_labels));

    Y = np.mat(np.zeros((m, num_labels)));

    for i in range(1, num_labels+1):
        pos_ = np.where(y == i)[0];
        Y[pos_,:] = np.repeat(E[i-1,:], len(pos_), 0);

    return Y;