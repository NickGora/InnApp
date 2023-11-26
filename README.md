# InnApp
Project "intent classification of user requests"

Project includes all scripts and data were used during implementation of app.

Files descriptions:

requrements.txt                             -  file with list needed libraries

eda.py                                      -  script for exploritary data analise

text_preprocessing.py                       -  script for preprocess text and labels in dataset, preprocessed data saved in csv

generate_syntetic_data.py                   -  scrip to gererate syntetic data, original and syntetic data saved toghether in csv

gscv_ada.py, lrcv.py, search_ann_params.py  -  scripts for searching best model hyperparamaters

model_train_eval.py                         -  scrip for training and evaluation models, results (bar chart) saved in png

best_model_train_save.py                    -  script for training and saving model with best performance in pickle file

app.py                                      -  scrip of window app named "InnApp"

*.png                                       -  pic with bar charts

*.ini                                       -  config files for app "InnApp" *** tokens.ini, clasess.ini should be read with utf-8

model.pkl                                   -  pickle file with save model

train.csv                                   -  file with training data
