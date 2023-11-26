# import modules
import os

import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# load data
path = os.getcwd()

x_b = pd.read_csv(f'{path}/X_bagged.csv').drop('Unnamed: 0', axis=1)
x_t = pd.read_csv(f'{path}/X_tokenized.csv').drop('Unnamed: 0', axis=1)
y = pd.read_csv(f'{path}/y.csv').drop('Unnamed: 0', axis=1)

# gs params
params = {'algorithm': ['SAMME'],
          'estimator': [DecisionTreeClassifier(max_depth=1),
                        DecisionTreeClassifier(max_depth=3),
                        DecisionTreeClassifier(max_depth=5)],
          'learning_rate': [0.1, 0.5, 1],
          'n_estimators': [10, 50, 100]}

# train and eval models
model = AdaBoostClassifier()
gs = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, 
                  cv=5, scoring='f1_macro', verbose=2)
gs.fit(x_b.to_numpy(), y.to_numpy().ravel())

# print best params
print('###############')
print('x bagged')
print(gs.best_params_)

# train and eval models
model = AdaBoostClassifier()
gs = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, 
                  cv=5, scoring='f1_macro', verbose=2)
gs.fit(x_t.to_numpy(), y.to_numpy().ravel())

# print best params
print('###############')
print('x tokenized')
print(gs.best_params_)

# params_ada = {'algorithm': 'SAMME', 
#               'estimator': DecisionTreeClassifier(max_depth=5), 
#               'learning_rate': 0.5, 
#               'n_estimators': 100}

