#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
sys.path.append('../')

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from pprint import pprint
import spacy as spacy
import pandas as pd
import numpy as np
import string
import re

# from ipynb.fs.full.BaseClassifierCNN import BaseClassifier


# In[2]:

with open("models/BaseClassifier.json", "r") as f:
    data = f.read()
    loaded_model = model_from_json(data)
loaded_model.load_weights("OPPClassifiers/BaseClassifier/base_classifier_weights.h5")

class Predictions(object):    
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=5000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
        
    def preprocess_text(self, df):
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
        brief_cleaning = (re.sub("[^A-Za-z]+", ' ', str(row)).lower() for row in df['Clauses'])
        text = [self.cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
        return text
    
    def cleaning(self, doc):
        txt = [token.lemma_ for token in doc if not token.is_stop]
        if len(txt) > 2:
            return ' '.join(txt)
        
    def remove_punct(self, text):
        text_nopunct = ''
        text_nopunct = re.sub('['+string.punctuation+']', '', text)
        text_nopunct = re.sub(r'\s+', ' ', text_nopunct)
        text_nopunct = re.sub(r'\d+', '', text_nopunct)  #remove numbers
        text_nopunct = text_nopunct.strip()              #remove whitespaces
        return text_nopunct
    
    def lower_token(self, tokens): 
        return [w.lower() for w in tokens]   
    
    def removeStopWords(self, tokens): 
        stoplist = stopwords.words('english')
        return [word for word in tokens if word not in stoplist]

    def text_to_vec(self, df, maxlen):
        clauses = df.Clauses.tolist()
        self.tokenizer.fit_on_texts(clauses)
        sequences = self.tokenizer.texts_to_sequences(clauses)
        clauses = pad_sequences(sequences, maxlen=maxlen, padding='post')
        return clauses

    def get_prediction(self, loaded_model, df, maxlen):
        df['text_clean'] = df['Clauses'].apply(lambda x: self.remove_punct(x))
        tokens = [word_tokenize(sen) for sen in df.text_clean]
        lower_tokens = [self.lower_token(token) for token in tokens]
        filtered_words = [self.removeStopWords(sen) for sen in lower_tokens]
        df['text_final'] = [' '.join(sen) for sen in filtered_words]
        df['tokens'] = filtered_words
        text = self.preprocess_text(df)
        df.Clauses = df.text_final
        df = df[['Clauses', 'tokens']]
        
        clauses = self.text_to_vec(df, maxlen)
        
        predictions = []
        for i in range(len(clauses)):
            pred = loaded_model.predict(np.expand_dims(clauses[i], 0))
            predictions.append(pred)
        return predictions


