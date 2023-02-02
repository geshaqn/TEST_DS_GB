from csv import reader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from random import randint
from sklearn.pipeline import Pipeline
import joblib

f = open('data.csv')
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
model = NuSVR(C = 1.0, nu = 0.1)
model.fit(train_X, train_Y)
pipe = Pipeline([('Scaler',scaler),('SVR',model)])
joblib.dump(pipe, "fitted_model.sav")
