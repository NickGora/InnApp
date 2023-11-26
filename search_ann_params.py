# import modules
import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch.nn as nn
from torch.optim import Adam
from torch import Tensor

from tqdm import tqdm

import gc


# load data
path = os.getcwd()

x_b = pd.read_csv(f'{path}/X_bagged.csv').drop('Unnamed: 0', axis=1)
#x_t = pd.read_csv(f'{path}/X_tokenized.csv').drop('Unnamed: 0', axis=1)
y = pd.read_csv(f'{path}/y.csv').drop('Unnamed: 0', axis=1)

# shuffle data
x_t_y = pd.concat([x_b, y], axis=1)
x_t_y = x_t_y.sample(frac=1)
x_t = x_t_y.drop('label', axis=1)
y = x_t_y['label']

del x_t_y

# convert numer labels to one-hot encoded labels
y = pd.get_dummies(y, columns=['label', ], dtype=int)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(x_t, y, test_size=0.3)

# conver to Tensors
x_train = Tensor(x_train.to_numpy())
y_train = Tensor(y_train.to_numpy())
x_test = Tensor(x_test.to_numpy())
y_test = Tensor(y_test.to_numpy())

# init models
# ALERT. I tried different types of model with emedining, gru, etc
# Models bellow are just and example of variations
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
model_1 = nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.Sigmoid(),
                         nn.Dropout(0.1))

model_2 = nn.Sequential(nn.Linear(input_dim, input_dim),
                          nn.Sigmoid(),
                          nn.Linear(input_dim, output_dim),                         
                          nn.Sigmoid())

model_3 = nn.Sequential(nn.Linear(input_dim, input_dim),
                          nn.Sigmoid(),
                          nn.Linear(input_dim, input_dim),
                          nn.Sigmoid(),
                          nn.Linear(input_dim, output_dim),                         
                          nn.Sigmoid())

models = [model_1, model_2, model_3]

# model hyperparameters
LR = 0.01
EPOCHS = 10
BATCH_SIZE = 100


# func of returning model name
def return_model_name():
    model_names = ['model_1', 'model_2', 'model_3']
    num = 0
    while num < len(model_names):
        yield model_names[num]
        num += 1


# create generator to return model name
model_name = return_model_name()

# models training
for model in models:
    optimizer = Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, x_train.shape[0]-BATCH_SIZE, BATCH_SIZE)):
            predictions = model(x_train[i:i+BATCH_SIZE])
            optimizer.zero_grad()
            loss=loss_func(predictions, y_train[i:i+BATCH_SIZE])
            loss.backward()
            optimizer.step()

            # free memory
            del predictions
            del loss
            
    Y_pred = model(x_train)
    Y_test_pred = model(x_test)

    f = f1_score

    print('####################')
    print(f'{next(model_name)}')

    print(f'CES_train: {loss_func(Y_pred, y_train)}, CES_test: {loss_func(Y_test_pred, y_test)}')

    # convert probabilities labels to numeric labels for F1 score
    Y_pred = np.argmax(Y_pred.detach().numpy(), axis=-1)
    y_f1_train = np.argmax(y_train, axis=-1)

    Y_test_pred = np.argmax(Y_test_pred.detach().numpy(), axis=-1)
    y_f1_test = np.argmax(y_test, axis=-1)

    print(f'F1_train: {f(Y_pred, y_f1_train, average="macro")}, F1_test: {f(Y_test_pred, y_f1_test, average="macro")}')

    # free memory
    del Y_pred
    del Y_test_pred

    gc.collect()

