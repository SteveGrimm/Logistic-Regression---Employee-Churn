from google.colab import drive
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

WD = '/content/Drive'
drive.mount(WD)

PATH = '/content/Drive/My Drive/Colab Notebooks/1000ml/Unit2/Project4/hr-data.csv'
PREDS_DEST = '/content/Drive/My Drive/Colab Notebooks/1000ml/Unit2/Project4/predictions.csv'
ESTIMATOR = Pipeline(steps = [('scaler', StandardScaler()), 
                              ('clf', LogisticRegression(penalty = 'l2', 
                                                        class_weight = 'balanced', 
                                                        solver = 'liblinear', 
                                                        random_state=42,
                                                        C = 1e-06))])
