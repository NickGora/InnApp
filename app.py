# modules for window managing
import tkinter as tk
from tkinter.messagebox import showwarning

# modules for file managing
import os
import configparser

# modules for text preprocessing
from pyaspeller import YandexSpeller

import sys
import pathlib
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    os.environ["PYMORPHY2_DICT_PATH"] = str(pathlib.Path('C:/Users/user/AppData/Local/Programs/Python/Python38/Lib/site-packages/pymorphy2_dicts_ru/data'))

from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import re

# modules for managing ml model
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

# window class
class Window(tk.Tk):
    def __init__(self, 
                 configs:dict, 
                 classes:dict, 
                 tokens:dict, 
                 model:LogisticRegression) -> None:
        super().__init__()

        # init variables to store params
        self.geometry_config = ''
        self.title_config = ''
        self.text_size_config = 0

        # unpack configs
        self._unpack_configs(configs)

        # set params of window 
        self.geometry(self.geometry_config)
        self.title(self.title_config)

        # config grid
        self._grid_config()

        # init tk variables
        self.label_var = tk.StringVar(self)
        self.entry_var = tk.StringVar(self)

        # create and pack widjets
        self._create_widgets()
        self._place_widgets()

        # set classes, tokens and model
        self.classes = classes
        self.model = model
        self.tokens = tokens

        # init preprocessing methods
        self._init_preprocessing()
    
    # method to unpack configs from dict
    def _unpack_configs(self, configs:dict) -> None:
        self.geometry_config = configs['geometry']
        self.title_config = configs['title']
        self.text_size_config = configs['text_size']

    # method to configure grid of window
    def _grid_config(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

    # method to create widgets
    def _create_widgets(self) -> None:
        self.label = tk.Label(self, 
                              text='Label', 
                              textvariable=self.label_var, 
                              font=self.text_size_config)
        self.entry = tk.Entry(self, 
                              textvariable=self.entry_var, 
                              font=self.text_size_config)
        self.buttom = tk.Button(self, 
                                text='Predict', 
                                command=self._buttom_func, 
                                font=self.text_size_config)

    # method to place widgets on window
    def _place_widgets(self) -> None:
        self.label.grid(row=0, column=0, sticky="nsew")
        self.entry.grid(row=1, column=0, sticky="nsew")
        self.buttom.grid(row=2, column=0, sticky="nsew")

    # method to init preprocessing methods
    def _init_preprocessing(self) -> None:
        self.speller = YandexSpeller()
        self.lemmatizer = MorphAnalyzer()
        self.stopwords = stopwords.words("russian")

    # method of button function
    def _buttom_func(self) -> None:
        text = self.entry_var.get()
        if text:
            text = self._text_preprocess(text)
            label = self._predict(text)
            label = self.classes[str(label)]
            self.label_var.set(label.upper())
        else:
            self.label_var.set('NONE')

    # method to preprocess text
    def _text_preprocess(self, text:str) -> list:
        text = text.lower()
        text = re.sub('[^а-яa-z]+', ' ', text)
        text = self.speller.spelled(text)
        text = [self.lemmatizer.parse(word)[0].normal_form for word in text.split()]
        text = [word for word in text if word not in self.stopwords]
        text = [self.tokens[word] 
                if word in self.tokens.keys() 
                else self.tokens['<pad>'] 
                for word in text]
        text = self._convert_to_bag(text)
        return text
    
    def _convert_to_bag(self, text) -> list:
        bag = [0] * len(self.tokens)
        for word in text:
            bag[int(word)] = 1
        return bag 

    # method to predict class of text
    def _predict(self, text:list) -> int:
        return self.model.predict(np.array(text).reshape(1, -1))[0]


# func to store default configs
def _default_config() -> dict:
    return {'geometry': '400x400',
            'title': 'InnApp',
            'text_size': 20}

# func to store default classes
def _default_classes() -> dict:
    return {0: 'FAQ - тарифы и услуги',
            1: 'мобильная связь - тарифы',
            2: 'Мобильный интернет',
            3: 'FAQ - интернет',
            4: 'тарифы - подбор',
            5: 'Баланс',
            6: 'Мобильные услуги',
            7: 'Оплата',
            8: 'Личный кабинет',
            9: 'SIM-карта и номер',
            10: 'Роуминг',
            11: 'запрос обратной связи',
            12: 'Устройства',
            13: 'мобильная связь - зона обслуживания'}

# func to check if file exist
def _exist(path:str) -> bool:
    return os.path.isfile(path)

# funct to show warnings
def _show_warning(text:str) -> None:
    TITLE_TEXT = 'FILE DOES NOT EXIST'
    text = f'file {text} does not exist\nDefault loaded'
    root = tk.Tk()
    root.withdraw()
    showwarning(title=TITLE_TEXT, message=text)
    root.destroy()

# read config file
def _read_config(path:str) -> dict:
    configs = {}
    if _exist(path):
        parser = configparser.ConfigParser()
        parser.read(path)
        configs = dict(parser['app'])
    else:
        _show_warning(path)
        configs = _default_config()
    return configs

# read classes file
def _read_classes(path:str) -> dict:
    classes = {}
    if _exist(path):
        parser = configparser.ConfigParser()
        parser.read(path, encoding='utf-8')
        classes = dict(parser['classes'])
    else:
        _show_warning(path)
        classes = _default_classes()
    return classes

# read tokens file
def _read_tokens(path:str) -> dict:
    tokens = {}
    if _exist(path):
        parser = configparser.ConfigParser()
        parser.read(path, encoding='utf-8')
        tokens = dict(parser['tokens'])
    else:
        TEXT = 'Without token file app will crash'
        _show_warning(f'{path}\n{TEXT}')
    return tokens

# read model file
def _read_model(path:str) -> LogisticRegression:
    model = None
    if _exist(path):
        with open(path, 'rb') as file:
            model = pickle.load(file)
    else:
        TEXT = 'Without model file app will crash'
        _show_warning(f'{path}\n{TEXT}')
    return model


# entry point of app
if __name__ == '__main__':
    # create file paths
    path_app = os.getcwd()
    path_configs = f'{path_app}/configs.ini'
    path_classes = f'{path_app}/classes.ini'
    path_tokens = f'{path_app}/tokens.ini'
    path_model = f'{path_app}/model.pkl'

    # read files
    configs = _read_config(path_configs)
    classes = _read_classes(path_classes)
    tokens = _read_tokens(path_tokens)
    model = _read_model(path_model)

    # create and run window
    root = Window(configs, classes, tokens, model)
    root.mainloop()
