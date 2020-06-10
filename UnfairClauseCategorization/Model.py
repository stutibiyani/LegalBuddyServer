#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import model_from_json
import numpy as np


# In[2]:


class FairnessClassifierModel(object):
    CATEGORY_LIST = ['Fair', 'Unfair']
    class_name = "FCM"
    
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as f:
            data = f.read()
            self.loaded_model = model_from_json(data)
        self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model._make_predict_function()
        
    def predict_class(self, sent):
        pred = self.loaded_model.predict(np.expand_dims(sent[0],0))
        # self.pred = self.loaded_model.predict(sent)
        return pred
        #return FairnessClassifierModel.CATEGORY_LIST[np.argmax(self.pred)]


# In[3]:


class UnfairClassifierModel(object):
    CATEGORY_LIST = ['Arbitration', 'Unilateral_Change', 'Content_Removal', 'Jurisdiction', 
                           'Choice_Of_Law', 'Limitation_Of_Liability', 'Unilateral_Termination', 'Contract_By_Using']
    class_name = "UCM"
    
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as f:
            data = f.read()
            self.loaded_model = model_from_json(data)
        self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model._make_predict_function()
        
    def predict_class(self, sent):
        pred = self.loaded_model.predict(np.expand_dims(sent[0],0))
        return pred
        # self.pred = self.loaded_model.predict(sent)
        # print(self.pred)
        # print(type(self.pred))
        # return UnfairClassifierModel.CATEGORY_LIST[np.argmax(self.pred)]
        # return self.pred[0]


# In[ ]:






# %%
