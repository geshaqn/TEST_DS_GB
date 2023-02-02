from csv import reader
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from random import randint
import joblib

f = open(r'D:\Mine\data.csv')
raw_data = list(reader(f))
title = raw_data[0][1:]
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
train_X = raw_data[:,1:]
train_Y = target
scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
model = SGDRegressor(max_iter=3000, tol=1e-4)
model.fit(train_X, train_Y)
joblib.dump(model, "fitted_model.sav")
