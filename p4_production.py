# -*- coding: utf-8 -*-
"""P4_Production.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ku8KiLyNjZ3zKnEJfhYAWACn6IVmIgsD
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from google.colab import drive
drive.mount('/content/Drive')

data = pd.read_csv('/content/Drive/My Drive/Colab Notebooks/1000ml/Unit2/Project4/hr-data.csv')

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

predictions.to_csv('/content/Drive/My Drive/Colab Notebooks/1000ml/Unit2/Project4/predictions.csv')
