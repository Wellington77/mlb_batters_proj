# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 00:26:51 2021

@author: welli
"""

import pandas as pd
import numpy as np

#importing data
df = pd.read_csv('mlb_data_cleaning.csv')
df.columns

#splitting into training and testing sets
from sklearn.model_selection import train_test_split

X = df[['G' , 'H' , 'Age' , 'PA', 'AB']]
y= df['BA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
predictions = regressor.predict(X_test)
print(predictions)

from sklearn import metrics
metrics.explained_variance_score(y_test,predictions)

#cross val score
from sklearn.model_selection import cross_val_score
val_score = np.mean(cross_val_score(regressor, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))
print(val_score)


#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))

#GrirdsearchCV, checking if this brings better score
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse', 'mae'), 'max_features':('auto','sqrt','log2')}
gs=GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

tpred_rg = regressor.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_rg)
mean_absolute_error(y_test, tpred_rf)
tpred_rg
tpred_rf


#saving model 
import pickle

pickle.dump(regressor,open('model.pkl', 'wb'))



