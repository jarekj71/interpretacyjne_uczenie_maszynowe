#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:48:01 2023

@author: jarekj
"""

#%%
import os
os.chdir(os.path.dirname(__file__))
import pandas as pd
import dice_ml
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

#%%
data = pd.read_csv("Datasets/diabetes_class.csv")
X = data.drop("Class",axis=1)
enc = LabelEncoder().fit(data.Class)
y = enc.transform(data.Class)

#%%
n_splits = 150
yhats = np.full((n_splits,y.shape[0]),np.nan)
ss = ShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=0)
for i, (train_index, test_index) in enumerate(ss.split(X)):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model = rfc().fit(X_train, y_train)
    yhat = model.predict(X_test)
    yhats[i][test_index] = yhat
#%%
zeros = (yhats==0).sum(axis=0)
ones = (yhats==1).sum(axis=0)

sums = zeros+ones
zeros_ratio = zeros/sums
ones_ratio = ones/sums

#%%
test_index = 37
X_train = X.drop(axis=0,index=test_index)
y_train = np.delete(y,test_index)
X_test = X.iloc[test_index:test_index+1]

model = rfc().fit(X_train, y_train)

m = dice_ml.Model(model=model, backend="sklearn")
d = dice_ml.Data(dataframe=data,continuous_features=data.columns.to_list()[:-1],outcome_name='Class')
exp = dice_ml.Dice(d, m, method="random")

#%%
e1 = exp.generate_counterfactuals(X_test, total_CFs=2, desired_class="opposite")
e0 = e1._cf_examples_list[0]
e0_df = e0.final_cfs_df

#%%
fig,ax = plt.subplots(figsize=(10,6))
ax.bar(x=X_test.columns,height=X_test.iloc[0])
ax.bar(x=X_test.columns,height=e0_df.iloc[0,:-1],width=0.6)
ax.tick_params(axis='x', labelrotation = 90)

#%%
data = pd.read_csv("Datasets/diabetes_reg.csv")
X = data.drop("progress",axis=1)
y = data.progress

#%%
lo = LeaveOneOut()
yhats = []
for i, (train_index, test_index) in enumerate(lo.split(X)):
    X_train = X.iloc[train_index]
    y_train = y[train_index]
    X_test = X.iloc[test_index]
    model = rfr().fit(X_train, y_train)
    yhats.append(model.predict(X_test))
yhats_1 = np.array(yhats).flatten()

#%%
n_splits = 150
yhats = np.full((n_splits,y.shape[0]),np.nan)
ss = ShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=0)
for i, (train_index, test_index) in enumerate(ss.split(X)):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model = rfr().fit(X_train, y_train)
    yhat = model.predict(X_test)
    yhats[i][test_index] = yhat

yhats_2_mean = np.nanmean(yhats,axis=0)
yhats_2_std = np.nanstd(yhats,axis=0)

#%% PLOT

fig,ax = plt.subplots(figsize=(10,8))
ax1 = ax.scatter(y,yhats_1)
ax2 = ax.errorbar(y,yhats_2_mean,yhats_2_std,fmt=".r")


#%% Regression
test_index = 37
X_train = X.drop(axis=0,index=test_index)
y_train = np.delete(y.values,test_index)
X_test = X.iloc[test_index:test_index+1]

model = rfr().fit(X_train, y_train)

yhat = model.predict(X_test)
y_test=y.values[test_index]

#%%


m = dice_ml.Model(model=model, backend="sklearn",model_type="regressor")
d = dice_ml.Data(dataframe=data,continuous_features=data.columns.to_list()[:-1],outcome_name='progress')
exp = dice_ml.Dice(d, m, method="random")

#%%
e1 = exp.generate_counterfactuals(X_test, total_CFs=2, desired_range=[250,260])
e0 = e1._cf_examples_list[0]
e0_df = e0.final_cfs_df

#%%
fig,ax = plt.subplots(figsize=(10,6))
ax.bar(x=X_test.columns,height=X_test.iloc[0])
ax.bar(x=X_test.columns,height=e0_df.iloc[0,:-1],width=0.6)
ax.tick_params(axis='x', labelrotation = 90)