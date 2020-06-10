#!/usr/bin/env python
# coding: utf-8

# In[178]:
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

# from ipynb.fs.full.UnfairClauseCategorization import UnfairClauseCategorization 
# from ipynb.fs.full.FairnessClassifierCNN import FairnessClassifierCNN
# from UnfairClauseCategorization import UnfairClauseCategorization
# from UnfairClauseCategorization import FairnessClassifierCNN 


# In[189]:


with open("models/FairnessClassifier.json", "r") as f:
    data = f.read()
    loaded_model_1 = model_from_json(data)
loaded_model_1.load_weights("UnfairClauseCategorization/fairness_model_weights.h5")

with open("models/UnfairClassifier.json", "r") as f:
    data = f.read()
    loaded_model_2 = model_from_json(data)
loaded_model_2.load_weights("UnfairClauseCategorization/categorize_model_weights.h5")

# fcc = FairnessClassifierCNN()
# ucc = UnfairClauseCategorization()

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


# In[214]:


# text = ["page end by implied mail to implied mail content and infringement content",
#         "limit you and fullest may discretion this linden if any time content all of the other",
#         "by using or accessing the services , you agree to become bound by all the terms and conditions of this agreement.",
#         "9gag , inc reserves the right to remove any subscriber content from the site , suspend or \
#          terminate subscriber 's right to use the services at any time , or pursue any other remedy or relief available \
#          to 9gag , inc and/or the site under equity or law, for any reason -lrb- including , but not limited to , \
#          upon receipt of claims or allegations from third parties or authorities relating to such subscriber content \
#          or if 9gag , inc is concerned that subscriber may have breached the immediately preceding \
#          sentence -rrb- , or for no reason at all . ",
#         "this policy and consent forms part of our website terms of use and as such it shall \
#          be governed by and construed in accordance with the laws of england and wales . ",
#         "these pages , the content and infrastructure of these pages , and the online reservation service \
#          provided on these pages and through the website are owned , operated and provided by booking.com b.v. \
#          and are provided for your personal , non-commercial use only , subject to the terms and conditions set out below . "]


# # In[215]:


# text_df = pd.DataFrame(text)
# text_df.columns = ['Clauses']


# # In[216]:


# pred = Predictions()


# # In[217]:


# predictions_fair = pred.get_prediction(loaded_model_1, text_df, 241)


# # In[218]:


# pprint(predictions_fair)


# # In[219]:


# for i in range(len(predictions_fair)):
#     if(predictions_fair[i][0][1].round() == 0):
#         print(text_df['Clauses'][i])
#         text_df = text_df.drop(i, axis=0)
# predictions_unfair = pred.get_prediction(loaded_model_2, text_df, 191)
# pprint(predictions_unfair)


# In[ ]:




