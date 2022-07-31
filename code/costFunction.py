from turtle import dot
import numpy as np;
from mapY import mapY;
from cv2 import log, multiply
from sigmoid import sigmoid;
from sigmoidGradient import sigmoidGradient;

def costFunction(all_theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    m = len(X);
    X = np.mat(X); # 5000*400
    y = np.mat(y); # 5000*1
    Y = mapY(y, num_labels); #5000*10
    cost = 0; # 损失
    all_theta = all_theta.T;

    theta1 = np.reshape(all_theta[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1)); # 25*401
    theta2 = np.reshape(all_theta[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1)); # 10*26

    X = np.hstack((np.ones((m, 1), dtype= int), X));# 5000*401

    theta1_ = theta1.copy();
    theta1_[:, 0] = 0;
    theta2_ = theta2.copy();
    theta2_[:, 0] = 0;
    theta1_ = np.power(theta1_, 2);
    theta2_ = np.power(theta2_, 2);

    # 前向传播
    z2 = np.dot(X, theta1.T); #5000*25 
    a2 = sigmoid(z2); 
    a2 = np.mat(np.hstack((np.ones((len(a2), 1), dtype= int), a2))); # 5000*26
    z3 = np.dot(a2, theta2.T); # 5000*10;
    a3 = sigmoid(z3);

    # 计算损失
    cost = -(multiply(Y, log(a3))+multiply(1-Y, log(1-a3)));
    cost = cost.reshape(cost.shape[0]*cost.shape[1]);
    cost = sum(cost)/m + lambda_/(2*m)*(np.sum(theta1_)+np.sum(theta2_));


    # 反向传播
    grad1 = np.mat(np.zeros((hidden_layer_size,input_layer_size+1))); # 25*401
    grad2 = np.mat(np.zeros((num_labels,hidden_layer_size+1))); # 10*26
    
    for i in range(m):
        a_1 = X[i, :]; # 1*401
        z_2 = np.dot(theta1, a_1.T); # 25*1
        a_2 = sigmoid(z_2);
        a_2 = np.r_[np.ones((1,1)) , a_2]; # 26*1
        z_3 = np.dot(theta2, a_2); 10*1
        a_3 = sigmoid(z_3);

        err3 = np.zeros((num_labels, 1));
        for k in range(num_labels):
            t = 0;
            if y[i] == k+1:
                t = 1;
            err3[k] = a_3[k] - t;

        err2 = np.dot(theta2.T, err3)[1:]; # 25*1 
        err2 = multiply(err2, sigmoidGradient(z_2));

        grad1 = grad1 + np.dot(err2, a_1); # 25*401
        grad2 = grad2 + np.dot(err3, a_2.T); # 10*26
    
    grad1 = 1/m*grad1;
    grad2 = 1/m*grad2;
    all_grad = np.mat(np.hstack((grad1.reshape(hidden_layer_size*(input_layer_size+1)), grad2.reshape(num_labels*(hidden_layer_size+1)))));

    return cost, all_grad;
