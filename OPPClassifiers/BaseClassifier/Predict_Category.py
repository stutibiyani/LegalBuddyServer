#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import numpy as np


# In[24]:


class Predict(object):
    
    def __init__(self, model_file, weights_file):
        with open(model_file, "r") as f:
            data = f.read()
            self.model = model_from_json(data)
        self.model.load_weights(weights_file)
        self.categories = ['Third Party Sharing/Collection', 'First Party Collection/Use', 'Other', 'User Choice/Control', 'Policy Change', 'Data Security', 'International and Specific Audiences', 'User Access, Edit and Deletion', 'Data Retention', 'Do Not Track']
    
    def text_to_vec(self, sentence):
        tokenizer = Tokenizer(num_words=5000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
        tokenizer.fit_on_texts(sentence)
        sequences_test = tokenizer.texts_to_sequences(sentence)
        x = pad_sequences(sequences_test, maxlen=150, padding='post')
        # print(tokenizer.sequences_to_texts(x))
        return x

    def get_prediction(self, sentences):
        indices = []
        for sentence in sentences:
            text_vec = self.text_to_vec(sentence)
            self.pred = self.model.predict(np.expand_dims(text_vec[0], 0))[0]
            indices.append(np.argmax(self.pred))
        return indices


# In[25]:


# base = Predict(
#     "Base Classifier.json",
#     "weights.h5"
# )


# # In[26]:


# result = base.get_prediction("You should be aware that if you voluntarily disclose personally identifiable information in an e-mail or other communications with third parties listed on the Sites or in other materials, that information, along with any other information disclosed in your communication, can be collected and correlated and used by such third parties and may result in your receiving unsolicited messages from other persons. Such collection, correlation, use and messages are beyond our control.  ")


# # In[27]:


# print(result)


# In[17]:





# In[ ]:




