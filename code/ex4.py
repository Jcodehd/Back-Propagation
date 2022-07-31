from cgi import test
from gettext import find
from mimetypes import init
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np;
import scipy.io as sc
import scipy.optimize as scio
from mapY import mapY;
from sigmoidGradient import sigmoidGradient;
from initializeTheta import initializeTheta;
from costFunction import costFunction;
from predict import predict;


# 设置分类数、 输入层大小、隐藏层大小
num_labels = 10;
input_layer_size = 400;
hidden_layer_size = 25;
lambda_ = 1;

# 加载数据
data = sc.loadmat('Machine Learning/Neural Network/Back Propagation/ex4data1.mat');
weight = sc.loadmat('Machine Learning/Neural Network/Back Propagation/ex4weights.mat');

init_theta1 = initializeTheta(input_layer_size, hidden_layer_size);
init_theta2 = initializeTheta(hidden_layer_size, num_labels);
init_theta = np.mat(np.hstack((init_theta1.reshape(hidden_layer_size*(input_layer_size+1)), init_theta2.reshape(num_labels*(hidden_layer_size+1)))));
X = data['X']; # 5000*400
y = data['y']; # 5000*1
Y = mapY(y, num_labels); # 5000*10


result = scio.fmin_tnc(func=costFunction, x0=init_theta, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_));

theta1 = np.reshape(result[0][:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1)); # 25*401
theta2 = np.reshape(result[0][hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1)); # 10*26

p = predict(theta1, theta2, X);

print('训练模型的精度为: ', np.mean(p==y)*100, '%');












