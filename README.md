# Unit2

During Project 4 I built a model to predict whether company employees were likely to churn or not.
To solve this Problem I built a Logistic Regression Model using

* Scaling, with Pipeline
* Grid cross validation

and the following libraries:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

My Model was able to correctly identify churning employees 75% of the time.
