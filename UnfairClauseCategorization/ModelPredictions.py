#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
from nltk.corpus import stopwords
import re
from UnfairClauseCategorization import Model
# from ipynb.fs.full.Model import FairnessClassifierModel as FCM
# from ipynb.fs.full.Model import UnfairClassifierModel as UCM


# In[2]:
loaded_model_1 = Model.FairnessClassifierModel("models/FairnessClassifier.json", "UnfairClauseCategorization/fairness_model_weights.h5")
loaded_model_2 = Model.UnfairClassifierModel("models/UnfairClassifier.json", "UnfairClauseCategorization/categorize_model_weights.h5")      

class Predictions(object):
    
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=5000,filters='!"#$%&()*+,-./:<;=?>@[\\]^_`{|}~\t\n\'',lower=True)
    
    # def clean_text(self):
    #     sentence = re.sub('[^a-zA-Z]', ' ', self.text)
    #     sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    #     sentence = re.sub(r'\s+', ' ', sentence)
    #     return sentence

    # def remove_stopwords(self):
    #     stop_words = set(stopwords.words('english'))
    #     no_stopword_text = [w for w in self.text.split() if not w in stop_words]
    #     return ' '.join(no_stopword_text)

    def text_to_vec(self, maxlen,text):
        self.tokenizer.fit_on_texts(text)
        sequences_test = self.tokenizer.texts_to_sequences(text)
        # tokenizer = Tokenizer(num_words=5000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
        #                   lower=True)
        # tokenizer.fit_on_texts(self.text)
        # sequences_test = tokenizer.texts_to_sequences(self.text)
        x = pad_sequences(sequences_test, maxlen=maxlen,padding = 'post')
        return x

    def get_prediction(self, loaded_model,text):
        # print(text)
        # shape_value = 0
        # self.text = self.clean_text()
        # self.text = self.remove_stopwords()
        # if (loaded_model.class_name == "FCM"):
        #     shape_value = 150
        # else:
        #     shape_value = 292
        text_vec = self.text_to_vec(150,text)
        ypred = loaded_model.predict_class(text_vec)
        return ypred