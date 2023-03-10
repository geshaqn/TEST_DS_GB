from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from random import randint
from sklearn import tree
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor

f = open('data.csv')
raw_data = list(reader(f))
title = raw_data[0]
raw_data = np.array(raw_data[1:], float)
data = []
target = np.array([0], float)
max_cycle = 0
cur_cycle = -1
for i in range(len(raw_data) - 1, -1, -1):
    if raw_data[i, 0] != cur_cycle:
        cur_cycle = raw_data[i, 0]
        max_cycle = raw_data[i, 1]
    target = np.insert(target, 0, raw_data[i, 1] / max_cycle)
target = target[:target.size - 1]
test_id = [randint(1,80) for i in range(8)]
train_X, test_X, train_Y, test_Y = [],[],[],[]
for i in range(len(raw_data)):
    if raw_data[i, 0] in test_id:
        test_X.append(raw_data[i, 1:])
        test_Y.append(target[i])
    else:
        train_X.append(raw_data[i, 1:])
        train_Y.append(target[i])
train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)
scaler = StandardScaler()
scaler.fit(raw_data[:, 1:])
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
model = tree.DecisionTreeRegressor()
model.fit(train_X, train_Y)
print("mse tree:", mean_squared_error(test_Y, model.predict(test_X)))
model = SGDRegressor(max_iter = 3000, tol = 1e-4)
model.fit(train_X, train_Y)
print("mse SGD:", mean_squared_error(test_Y, model.predict(test_X)))
model = NuSVR(C = 1.0, nu = 0.1)
model.fit(train_X, train_Y)
print("mse SVM:", mean_squared_error(test_Y, model.predict(test_X)))
model = MLPRegressor(random_state=randint(0,10000),max_iter=300)
model.fit(train_X, train_Y)
print("mse NN:", mean_squared_error(test_Y, model.predict(test_X)))
