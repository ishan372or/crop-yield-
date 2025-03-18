import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d=pd.read_csv("cropyield.csv")
x=d.iloc[:, :-1].values
y=d.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_predlr=lr.predict(x_test)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=0,max_depth=20,min_samples_leaf=2)
rf.fit(x_train,y_train)
y_predrf=rf.predict(x_test)

from sklearn.metrics import r2_score
print("multinear regression model:",r2_score(y_test,y_predlr))
print("random forest regression model:",r2_score(y_test,y_predrf))
