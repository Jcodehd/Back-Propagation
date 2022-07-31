import numpy as np;
from sigmoid import sigmoid;

def predict(theta1, theta2, X):
    X = np.mat(X);
    theta1 = np.mat(theta1);
    theta2 = np.mat(theta2);

    # 样本容量
    m = len(X);

    # 扩增
    X = np.hstack((np.ones((m,1), dtype= int),X));

    # 前向传播
    a2 = sigmoid(np.dot(X, theta1.T));

    a2 = np.hstack((np.ones((m,1), dtype= int),a2));

    a3 = sigmoid(np.dot(a2, theta2.T));

    p = np.where(a3 == np.max(a3, axis=1))[1]+1;

    p = np.mat(p).T;

    return p;


