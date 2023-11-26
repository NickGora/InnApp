# import modules
import os

import pandas as pd

import numpy as np

from tqdm import tqdm


# read data
path = os.getcwd()
x_b =  pd.read_csv(f'{path}/X_bagged.csv').drop('Unnamed: 0', axis=1)
x_t =  pd.read_csv(f'{path}/X_tokenized.csv').drop('Unnamed: 0', axis=1)
y =  pd.read_csv(f'{path}/y.csv').drop('Unnamed: 0', axis=1)

# compute classes destribution
destributions = y.value_counts() / y.shape[0]

# delete classes with distribution higher 0.1
thrfld = 0.1
classes = [(x[0], y) for x,y in zip(destributions.index, destributions.values) if y<thrfld]


# func to create word set
def word_set(text:pd.Series) -> np.array:
    words = []
    for line in text.to_numpy():
        for word in line:
            if word not in words:
                words.append(word)
    return np.array(words)


# compute total amount of needed to generate data
n_smpales = 0
y_t_b = np.array([])
for cls, dstrbtn in tqdm(classes):
    n = int((thrfld-dstrbtn)*y.shape[0])
    y_t_b = np.concatenate([y_t_b, [cls]*n])
    n_smpales += n

# create matrix for bag of words
arr_x_t = []
arr_x_b = np.zeros([n_smpales, x_b.shape[1]])

# create word sets for each class
words_dict = {}
for cls, _ in tqdm(classes):
    temp = x_t.loc[y[y==cls].index]
    words = word_set(temp)
    words_dict[cls] = words

# create array of sentenses cutoff 
cutoffs = np.random.randint(3, 10, len(y))

# generate samples
for indx, cls, cutoff in tqdm(zip(range(len(y_t_b)), y_t_b, cutoffs)):
    words = words_dict[cls]
    np.random.shuffle(words)

    line = words[:cutoff]

    line_x_t = np.concatenate([line, [4964]*(x_t.shape[1]-len(line))])

    arr_x_t.append(line_x_t)

    for word in line:
        arr_x_b[word] = 1
    arr_x_b[4964] = 1

# print shapes to make sure it is okay
print(len(arr_x_t), len(arr_x_t[0]))
print(arr_x_b.shape)
print(y_t_b.shape)

# concat original and syntetic datasets
x_b = pd.concat([x_b, pd.DataFrame(arr_x_b, columns=x_b.columns)], axis=0)
x_t = pd.concat([x_t, pd.DataFrame(arr_x_t, columns=x_t.columns)], axis=0)
y = pd.concat([y, pd.DataFrame(y_t_b, columns=y.columns)], axis=0)

# save datasets
x_b.to_csv(f'{path}/X_bagged_extended.csv')
x_t.to_csv(f'{path}/X_tokenized_extended.csv')
y.to_csv(f'{path}/y_extended.csv')
