#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:42:13 2023

@author: jarekj
https://archive.ics.uci.edu/ml/datasets.php
"""
#%%
import os
os.chdir(os.path.dirname(__file__))
from sklearn.tree import DecisionTreeRegressor as TreeReg
from sklearn import tree
import numpy as np
from sklearn.linear_model import LinearRegression as ls
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data = pd.read_csv("Datasets/concrete.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]


#%%
model = TreeReg(max_depth=4).fit(X,y)
path = model.cost_complexity_pruning_path(X,y)
nodes = model.apply(X)

dfa = pd.DataFrame({"Nodes":nodes,"Concrete":y})
#%%
fig,ax=plt.subplots(figsize=(16,10),dpi=72)
tree.plot_tree(model,feature_names=X.columns)
fig.savefig("tree.pdf")

#%%
sns.boxplot(dfa,x="Nodes",y="Concrete")

#%% Różne 
selector = (nodes==5)
X1=X.iloc[selector]
y1 = y[selector]
model = ls().fit(X1,y1)
model.coef_
#%%
model = ls().fit(X,y)
model.coef_
