import pandas as pd

import matplotlib
matplotlib.use('Agg')

import pickle

d = pd.read_csv("cropyield.csv.csv")
x = d.iloc[:, :-1].values
y=d.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import  RandomForestRegressor

clf = RandomForestRegressor(n_estimators=10,random_state=0)
clf.fit(x_train,y_train)

pickle.dump(clf, open("model.pkl", "wb"))