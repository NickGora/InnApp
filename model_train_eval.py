# import modules
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch.nn as nn
from torch.optim import Adam
from torch import Tensor

from tqdm import tqdm


# load data
path = os.getcwd()
x_b = pd.read_csv(f'{path}/X_bagged.csv').drop('Unnamed: 0', axis=1)
x_t = pd.read_csv(f'{path}/X_tokenized.csv').drop('Unnamed: 0', axis=1)
y = pd.read_csv(f'{path}/y.csv').drop('Unnamed: 0', axis=1)


################# TRAINING MODELS PART #################


# split data
TEST_SIZE = 0.3
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_b, y, 
                                                            test_size=TEST_SIZE, 
                                                            shuffle=True)
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_t, y, 
                                                            test_size=TEST_SIZE, 
                                                            shuffle=True)

# train ada
# params
params_ada = {'algorithm': 'SAMME',
              'base_estimator': DecisionTreeClassifier(max_depth=5),
              'learning_rate':1,
              'n_estimators': 100}

model_ada_b = AdaBoostClassifier(**params_ada)
model_ada_b.fit(x_train_b.to_numpy(), y_train_b.to_numpy().ravel())

model_ada_t = AdaBoostClassifier(**params_ada)
model_ada_t.fit(x_train_t.to_numpy(), y_train_t.to_numpy().ravel())

# train lr
# params
params_lr = {'penalty': 'l2',
             'solver': 'lbfgs',
             'max_iter': 10000}

model_lr_b = LogisticRegression(**params_lr)
model_lr_b.fit(x_train_b.to_numpy(), y_train_b.to_numpy().ravel())

model_lr_t = LogisticRegression(**params_lr)
model_lr_t.fit(x_train_t.to_numpy(), y_train_t.to_numpy().ravel())

# train and eval ann
# convert numer labels to one-hot encoded labels for ann
y_ann = pd.get_dummies(y, columns=['label', ], dtype=int)

# split data for ann
TEST_SIZE = 0.3
x_train_b_ann, x_test_b_ann, y_train_b_ann, y_test_b_ann= train_test_split(x_b, y_ann, 
                                                            test_size=TEST_SIZE, 
                                                            shuffle=True)
x_train_t_ann, x_test_t_ann, y_train_t_ann, y_test_t_ann = train_test_split(x_t, y_ann, 
                                                            test_size=TEST_SIZE, 
                                                            shuffle=True)

# model hyperparameters
input_dim_b = x_train_b_ann.shape[1]
output_dim_b = y_train_b_ann.shape[1]
input_dim_t = x_train_t_ann.shape[1]
output_dim_t = y_train_t_ann.shape[1]
DROPOUT_PROB = 0.1
LR = 0.01
EPOCHS = 10
BATCH_SIZE = 100


# func to train ann models
def training_func(model:nn.Sequential, data_train:tuple) -> nn.Sequential:
    x_train, y_train = data_train
    optimizer = Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for _ in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, x_train.shape[0]-BATCH_SIZE, BATCH_SIZE)):
            predictions = model(x_train[i:i+BATCH_SIZE])
            optimizer.zero_grad()
            loss=loss_func(predictions, y_train[i:i+BATCH_SIZE])
            loss.backward()
            optimizer.step()

            # free memory
            del predictions
            del loss
               
    return model


# init model (bagged dataset)
model_ann_b = nn.Sequential(nn.Linear(input_dim_b, output_dim_b),
                         nn.Sigmoid(),
                         nn.Dropout(DROPOUT_PROB))

# train model (bagged dataset)
data_train_ann_b = (Tensor(x_train_b_ann.to_numpy()), Tensor(y_train_b_ann.to_numpy()))
model_ann_b = training_func(model_ann_b, data_train_ann_b)

# init mode l(tokenized dataset)
model_ann_t = nn.Sequential(nn.Linear(input_dim_t, output_dim_t),
                         nn.Sigmoid(),
                         nn.Dropout(DROPOUT_PROB))

# train model (tokenized dataset)
data_train_ann_t = (Tensor(x_train_t_ann.to_numpy()), Tensor(y_train_t_ann.to_numpy()))
model_ann_t = training_func(model_ann_t, data_train_ann_t)


################# EVALUATION MODELS PART #################


# compute f1 metrics for each model and dataset
f1_ada_b_train = f1_score(model_ada_b.predict(x_train_b), y_train_b, average='macro')
f1_ada_b_test = f1_score(model_ada_b.predict(x_test_b), y_test_b, average='macro')
f1_ada_t_train = f1_score(model_ada_t.predict(x_train_t), y_train_t, average='macro')
f1_ada_t_test = f1_score(model_ada_t.predict(x_test_t), y_test_t, average='macro')

f1_lr_b_train = f1_score(model_lr_b.predict(x_train_b), y_train_b, average='macro')
f1_lr_b_test = f1_score(model_lr_b.predict(x_test_b), y_test_b, average='macro')
f1_lr_t_train = f1_score(model_lr_t.predict(x_train_t), y_train_t, average='macro')
f1_f1_t_test = f1_score(model_lr_t.predict(x_test_t), y_test_t, average='macro')

# compute f1 for ann
# convert test dataset to Tensor
data_test_ann_b = (Tensor(x_test_b_ann.to_numpy()), Tensor(y_test_b_ann.to_numpy()))
data_test_ann_t = (Tensor(x_test_t_ann.to_numpy()), Tensor(y_test_t_ann.to_numpy()))

# predict
pred_train_b = model_ann_b(data_train_ann_b[0]).detach().numpy()
pred_test_b = model_ann_b(data_test_ann_b[0]).detach().numpy()
pred_train_t = model_ann_t(data_train_ann_t[0]).detach().numpy()
pred_test_t = model_ann_t(data_test_ann_t[0]).detach().numpy()

# convert probabilities labels to numeric labels for F1 score
pred_train_b = np.argmax(pred_train_b, axis=-1)
pred_test_b = np.argmax(pred_test_b, axis=-1)
pred_train_t = np.argmax(pred_train_t, axis=-1)
pred_test_t = np.argmax(pred_test_t, axis=-1)

y_train_b_ann = np.argmax(y_train_b_ann.to_numpy(), axis=-1)
y_test_b_ann = np.argmax(y_test_b_ann.to_numpy(), axis=-1)
y_train_t_ann = np.argmax(y_train_t_ann.to_numpy(), axis=-1)
y_test_t_ann = np.argmax(y_test_t_ann.to_numpy(), axis=-1)

# compute f1 for ann
f1_ann_b_train = f1_score(pred_train_b, y_train_b_ann, average='macro')
f1_ann_b_test = f1_score(pred_test_b, y_test_b_ann, average='macro')
f1_ann_t_train = f1_score(pred_train_t, y_train_t_ann, average='macro')
f1_ann_t_test = f1_score(pred_test_t, y_test_t_ann, average='macro')

# plot f1 metrics of bagged data on bar chart
fntsz = 20
name = ['ADA', 'LR', 'ANN']
x = np.arange(len(name))
plt.bar(x-0.1, [f1_ada_b_train, f1_lr_b_train, f1_ann_b_train], 0.2, color='blue')
plt.bar(x+0.1, [f1_ada_b_test, f1_lr_b_test, f1_ann_b_test], 0.2, color='red')
plt.title('Bag of Words Dataset', fontsize=fntsz+2)
plt.xticks(x, name, fontsize=fntsz)
plt.ylabel('F1 Score', fontsize=fntsz)
plt.legend(['Train', 'Test'], fontsize=fntsz)
plt.show()

# plot f1 metrics of tokenized data on bar chart
plt.bar(x-0.1, [f1_ada_t_train, f1_lr_t_train, f1_ann_t_train], 0.2, color='blue')
plt.bar(x+0.1, [f1_ada_t_test, f1_f1_t_test, f1_ann_t_test], 0.2, color='red')
plt.title('Tokenized Dataset', fontsize=fntsz+2)
plt.xticks(x, name, fontsize=fntsz)
plt.ylabel('F1 Score', fontsize=fntsz)
plt.legend(['Train', 'Test'], fontsize=fntsz)
plt.show()
