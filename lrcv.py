# import modules
import os

import pandas as pd

from sklearn.linear_model import LogisticRegressionCV


# load data
path = os.getcwd()

x_b = pd.read_csv(f'{path}/X_bagged.csv').drop('Unnamed: 0', axis=1)
x_t = pd.read_csv(f'{path}/X_tokenized.csv').drop('Unnamed: 0', axis=1)
y = pd.read_csv(f'{path}/y.csv').drop('Unnamed: 0', axis=1)

# train and eval models
model = LogisticRegressionCV(cv=5, max_iter=10000, scoring='f1_macro', verbose=2, n_jobs=2).fit(x_t, y.to_numpy().ravel())
print('###############')
print('x tokenized')
print(model.get_params())

model = LogisticRegressionCV(cv=5, max_iter=10000, scoring='f1_macro', verbose=2, n_jobs=2).fit(x_b, y.to_numpy().ravel())
print('###############')
print('x bagged')
print(model.get_params())

# lg_params_tokenized = {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 10000}
