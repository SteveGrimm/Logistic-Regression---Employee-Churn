# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('hr-data.csv')

#set variables
data['salary'] = np.where(data['salary'].isin(['low','medium']), 1, 0)
data['sales'] = np.where(data['sales'].isin(['RandD','management']), 0, 1)
x = data.drop(['id','churn'],axis=1)
y = data['churn']

#Make pipe, but change solver
estimator = Pipeline(steps = [('scaler', StandardScaler()), 
                              ('clf', LogisticRegression(penalty = 'l2', 
                                                        class_weight = 'balanced', 
                                                        solver = 'liblinear', 
                                                        random_state=42,
                                                        C = 1e-06))])

estimator.fit(x_train,y_train)

predictions = pd.DataFrame(estimator.predict(x))

predictions.to_csv('predictions.csv')

