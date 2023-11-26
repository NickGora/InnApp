# import modules
import os

import pandas as pd

import numpy as np

from pymorphy2 import MorphAnalyzer

from pyaspeller import YandexSpeller

from nltk.corpus import stopwords

from tqdm import tqdm

from configparser import ConfigParser


# set to show tqdm for pandas df
tqdm.pandas()

# read data
path = os.getcwd()
df = pd.read_csv(f'{path}/train.csv')

# map string labels to number labels
labels = dict((x,y) for x,y in zip(df['label'].unique(), np.arange(len(df['label'].unique()))))
df['label'] = df['label'].map(labels)

# func to preprocess texts
SYMBOLS = "'';:.,\[]{}<>?/!@#$%^&*()_+=-»«"

corrector = YandexSpeller()

lemmatizer = MorphAnalyzer()

stopwords = stopwords.words("russian")


def preprocess_text(line: str) -> str:
    clean_line = ''
    for word in str(line).split():
        word = word.lower()
        word = word.strip(SYMBOLS)
        word = corrector.spelled(word)
        word = lemmatizer.parse(word)[0].normal_form
        if word not in stopwords:
            clean_line += f'{word} '
    return clean_line


# preprocess texts
df['text'] = df['text'].progress_apply(preprocess_text)

# create tokens
tokens = df['text'].apply(lambda x: np.array(str(x).split())).to_numpy()
tokens = list(set(np.concatenate(tokens)))
tokens.append('<pad>')
tokens = dict((x,y) for x,y in zip(tokens, np.arange(len(tokens))))

# func to tokinise texts
def text_tokinization(line:str) -> list:
    tokenized = []
    for word in str(line).split():
        tokenized.append(tokens[word])
    return tokenized

# tokinise texts
df['text_tokinezed'] = df['text'].progress_apply(text_tokinization)

# find the longest text
longest = -1
for text in df['text_tokinezed']:
    if len(text) > longest:
        longest = len(text)

# func to append <pad> token to pad text lenght
def pad_text(line:list) -> list:
    while len(line) < longest:
        line.append(tokens['<pad>'])
    return line

# pad text
df['text_tokinezed'] = df['text_tokinezed'].progress_apply(pad_text)

# convert df of list to df of columns
X_df = pd.DataFrame(columns=np.arange(len(df['text_tokinezed'].loc[0])))
for indx, value in tqdm(enumerate(df['text_tokinezed'])):
    X_df.loc[indx] = value

# create bag of words
x_matrix = len(tokens)
y_matrix = df['text_tokinezed'].shape[0]
X_baged = np.array([0]*x_matrix*y_matrix).reshape(y_matrix, x_matrix)
for indx, value in tqdm(enumerate(df['text_tokinezed'])):
    last_seen = -1
    for v in value:
        if last_seen == v:
            break
        X_baged[indx][v] = 1
        last_seen = v

# print shape to make sure it is okay
print(X_baged.shape, len(tokens), df['text_tokinezed'].shape)

# save dataset for later use
pd.DataFrame(X_baged).to_csv(f'{path}/X_bagged.csv')
X_df.to_csv(f'{path}/X_tokenized.csv')
df['label'].to_csv(f'{path}/y.csv')

# save classes 
labels = dict([(v, k) for k, v in zip(labels.keys(), labels.values())])
parser = ConfigParser()
parser['classes'] = labels
with open(f'{path}/classes.ini', 'w', encoding='utf-8') as configfile:
    parser.write(configfile)

# save tokens
parser = ConfigParser()
parser['tokens'] = tokens
with open(f'{path}/tokens.ini', 'w', encoding='utf-8') as configfile:
    parser.write(configfile)
