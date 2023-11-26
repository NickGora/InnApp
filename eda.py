# import modules
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


# load data
path = os.getcwd()
df = pd.read_csv(f'{path}/train.csv')

# print dataset info 
print(df.info())

""" 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32800 entries, 0 to 32799
Data columns (total 3 columns):
#   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   id      32800 non-null  int64 
 1   text    32709 non-null  object 
 2   label   32800 non-null  object  
 dtypes: int64(1), object(2)
memory usage: 768.9+ KB 


Dataset has 3 columns and 32800 rows. 
Dataset uses 768.9 KB memory.
Column text contains non-fill requests (empty strings) 
"""

# print head
print(df.head())

""" 
None
id                                     text                     label 
0   0                     Как отключить тариф?     FAQ - тарифы и услуги 
1   1                                    тариф  мобильная связь - тарифы 
2   2                                    тариф  мобильная связь - тарифы 
3   3  Здрасте я хотел получить золотую карту      FAQ - тарифы и услуги                                                
4   4                            Золотую карту     FAQ - тарифы и услуги  


Dataset contains stop words, misspellings, punktuation marks

"""

# compute class destribution
destributions = df['label'].value_counts() / df.shape[0]
counts = df['label'].value_counts()

# show bar char with destributions and count
fntsz = 20
x = np.arange(destributions.shape[0])
#plt.bar(x, destributions.values, 0.2, color='blue')
plt.bar(x, counts.values, 0.2, color='red')
plt.title('Class Count', fontsize=fntsz+2)
plt.xticks(x, fontsize=fntsz, rotation=90)
plt.ylabel('Count', fontsize=fntsz)
#plt.legend(['Destribution', 'Count'], fontsize=fntsz)
plt.show()

""" 
Dataset has skew in classes 
""" 

# check if text contains words in English alphabet
english_words = [word for word in df['text'] if str(word).isascii() and str(word).isalpha()]
print(english_words[:10])
print(f'Destributions: {len(english_words)/df.shape[0]}')

"""
[nan, nan, nan, 'USSD', nan, nan, nan, nan, nan, 'Tarif']                                                               
Destributions: 0.0033841463414634146 


Dataset contains russian word written in english alphabet.
Destribution of with words is 0.003 (ingluding 'nan's). 
"""
