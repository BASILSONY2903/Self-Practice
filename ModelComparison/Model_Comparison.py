# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:41:52 2018

@author: basil.p.sony
"""

# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import random
from scipy.stats import randint as sp_randint
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import os
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import scipy
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt  
 
#Importing Data
INPUT_PATH = "C:/Users/basil.p.sony/Desktop/Python Stuff/Python/UCL_Data.csv"
dataset = pd.read_csv(INPUT_PATH)

# Data Treatment
print dataset.describe()
dataset['BareNuclei'] = dataset['BareNuclei'].str.replace('?','99')
median_value = dataset['BareNuclei'].median(skipna=True)
dataset['BareNuclei']=dataset.BareNuclei.mask(dataset.BareNuclei == 99,median_value)
dataset['BareNuclei'] = dataset.BareNuclei.fillna(median_value) 
#dataset = dataset.drop("BareNuclei", 1)


# Split data into Test and Train
X = dataset.iloc[:,0:9]
Y = dataset.iloc[:,9]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


#############################################Different Models################################################

###################### Decison Trees##################

### Grid Search####
pipeline = Pipeline([('clf',DecisionTreeClassifier(criterion = 'entropy'))])
parameters = {'clf__max_depth' : (50,100,150), 
              'clf__min_samples_leaf' : (1,2,3),
              'clf__min_samples_split' : (2,3)}
grid_search  = GridSearchCV(pipeline, parameters, n_jobs = -1 , verbose = 1, scoring = 'accuracy')
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print 'Best Score :', grid_search.best_score_
print 'Best Parameter set :'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score(y_test,y_pred)
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)

### Random Search####
pipeline = Pipeline([('clf',DecisionTreeClassifier(criterion = 'entropy'))])
parameters = {'clf__max_depth' : sp_randint(10,100), 
              'clf__min_samples_leaf' : sp_randint(2,6),
              'clf__min_samples_split' : sp_randint(2,8)}
model = RandomizedSearchCV(pipeline, param_distributions=parameters, n_jobs=1, cv=10, scoring='accuracy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print 'Best Score :', model.best_score_
print 'Best Parameter set :'
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score(y_test,y_pred)
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)



######################## Random Forest ##################

### Grid Search####
rfr = RandomForestClassifier(n_jobs=1, random_state=0)
param_grid = {'n_estimators': [10,20,30,40,50], 'max_features': [3,6,9], 'max_depth':[5,10,15,20]
,  'min_samples_leaf': [2,4,6], 'min_samples_split': [2,4,7]}
model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring='accuracy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print 'Best Score :', model.best_score_
print 'Best Parameter set :'
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score(y_test,y_pred)
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)


### Random Search####
rfr = RandomForestClassifier(n_jobs=1, random_state=0)
param_grid = {'n_estimators': sp_randint(10,100)
                , 'max_features': sp_randint(3,9)
                , 'max_depth':sp_randint(5,30)
                ,  'min_samples_leaf': sp_randint(2,6)
                , 'min_samples_split': sp_randint(2,8)
                }
model = RandomizedSearchCV(estimator=rfr, param_distributions=param_grid, n_jobs=1, cv=10, scoring='accuracy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print 'Best Score :', model.best_score_
print 'Best Parameter set :'
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score(y_test,y_pred)
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)




######################### XGBoost #########################

### Grid Search####
xgb = XGBClassifier(n_jobs=1, random_state=0)
param_grid = {'n_estimators': [20,30,40,50], 'learning_rate': [0.2,0.4,0.6], 'max_depth':[5,10,15,20,25]}
model = GridSearchCV(estimator=xgb, param_grid=param_grid, n_jobs=1, scoring='accuracy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print 'Best Score :', model.best_score_
print 'Best Parameter set :'
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score(y_test,y_pred)
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)

### Random Search####
rfr = XGBClassifier(n_jobs=1, random_state=0)
param_grid = {'n_estimators': sp_randint(10,100)
                , 'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                , 'max_depth':sp_randint(5,30)
                }
model = RandomizedSearchCV(estimator=rfr, param_distributions=param_grid, n_jobs=1, cv=10, scoring='accuracy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print 'Best Score :', model.best_score_
print 'Best Parameter set :'
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score(y_test,y_pred)
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)





######################### Logistic Regression #########################

### Grid Search####
logistic_regression_model = LogisticRegression()
param_grid = {
            'penalty':['l2'],
            'C':[1,10,100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],}
model = GridSearchCV(logistic_regression_model, param_grid,cv=10,verbose=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score_train = model.score(X_train, y_train)
accuracy_score_test = model.score(X_test, y_test)
print 'Best Score :', model.best_score_
print 'Best Parameter set :'
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score_test
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)

### Random Search####
logistic_regression_model = LogisticRegression()
param_grid = {
            'penalty':['l2'],
            'C':sp_randint(1,100),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],}
model = RandomizedSearchCV(estimator=logistic_regression_model, param_distributions=param_grid, n_jobs=1, cv=10, scoring='accuracy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score_train = model.score(X_train, y_train)
accuracy_score_test = model.score(X_test, y_test)
print 'Best Score :', model.best_score_
print 'Best Parameter set :'
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print ('\t%s : %r' % (param_name, best_parameters[param_name]))
print 'Confusion Metrics on test data :', confusion_matrix(y_test,y_pred)
print 'Test Accuracy :', accuracy_score_test
print 'Precision Recall f1 max :', classification_report(y_test,y_pred)


######################### SVM  #########################

C = 1.0  # SVM regularization parameter
# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)


