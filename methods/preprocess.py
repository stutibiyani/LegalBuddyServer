import sys
sys.path.append('../')
import string
import pandas as pd
import numpy as np
from pprint import pprint
from UnfairClauseCategorization import ModelPredictionsUnfair    #importing class
from OPPClassifiers.BaseClassifier import ModelPredictionsBaseClassifier
pd.options.display.max_colwidth = 10

text = "If you acquired the application in the United States or Canada, the laws of the state or province where you live (or, if a business, where your principal place of business is located) govern the interpretation of these terms, claims for breach of them, and all other claims (including consumer protection, unfair competition, and tort claims), regardless of confict of laws principles. Blizzard Entertainment is liable in accordance with statutory law (i) in case of intentional breach, (ii) in case of gross negligence, (iii) for damages arising as result of any injury to life, limb or health or (iv) under any applicable product liability act. Academiadotedu reserves the right, at its sole discretion, to discontinue or terminate the Site and Services and to terminate these Terms, at any time and without prior notice. As described above, you may have a relationship with one or more of our Partners, in which case we may share certain information with them in order to coordinate with them on providing the Netflix service to members and providing information about the availability of the Netflix service. Data is stolen. Userd id distributed to third party for money. Data can be given to Modiji as per request."

def pre_process_driver(data):   #Data coming from server
    #raw_data = data.split('. ')
    raw_data = [x.strip() for x in data.split('. ')]
    raw_data_df = pd.DataFrame(raw_data)
    raw_data_df.columns = ['Clauses']
    raw_data_df['Original_Clauses'] = raw_data_df['Clauses']
    
    server_response = {
        'Fairness':{
            'Arbitration': [],
            'Unilateral_Change': [],
            'Content_Removal':[],
            'Jurisdiction':[],
            'Choice_Of_Law':[],
            'Limitation_Of_Liability':[],
            'Unilateral_Termination':[],
            'Contract_By_Using':[]

        },
        'Categories':{
            'FirstPartyCollection':[],
            'ThirdPartySharingCollection':[],
            'DataSecurity':[],
            'Other':[],
            'UserChoiceControl':[],
            'PolicyChange':[],
            'DataRetention':[],
            'UserAccessEditandDeletion':[],
            'InternationalAndSpecificAudience':[],
            'DoNotTrack':[]
        }

    }
    '''
    raw_data = raw_data.lower() #convert to lowercase

    raw_data = raw_data.translate(str.maketrans('','',string.punctuation))  #remove punctation marks

    raw_data = processing_data(raw_data)   #calling remove_stopwords function 

    '''
    # Testing with models

    # 1: Fairness Classification

    fairness_prediction_object = ModelPredictionsUnfair.Predictions()  #creating class object and passing data to it.
    fair = fairness_prediction_object.get_prediction(ModelPredictionsUnfair.loaded_model_1,raw_data_df,241) 
    
    #removing fair clauses from the dataframe and further passing to unfair model
    fairness_dataframe_clone = raw_data_df.copy()
    for i in range(len(fair)):
        if(fair[i][0][1].round() == 0):
            fairness_dataframe_clone = fairness_dataframe_clone.drop(i,axis = 0)  #fair clauses dropped from the dataframe

    unfair = fairness_prediction_object.get_prediction(ModelPredictionsUnfair.loaded_model_2,fairness_dataframe_clone,191)
    unfair = np.round(unfair)
    # pprint(fairness_dataframe_clone)
    fairness_dataframe_clone = fairness_dataframe_clone['Original_Clauses'].tolist()    #converting DF back to List
    text_index = 0
    for i in range(0,len(unfair)):
        for length in range(0,8):
            if(unfair[i][0][length]) == 1:
                unfair_key = FairnessKeys(length)
                server_response['Fairness'][unfair_key].append(fairness_dataframe_clone[text_index])
        text_index = text_index + 1   

    # 2: Category Classification
    
    category_prediction_object = ModelPredictionsBaseClassifier.Predictions()
    values = category_prediction_object.get_prediction(ModelPredictionsBaseClassifier.loaded_model,raw_data_df,214)
    values = np.round(values)
    pprint(values)
    text_index = 0
    for i in range(0,len(values)):
        for category_length in range(0,10):
            if values[i][0][category_length] == 1:
                category_key = CategoryKeys(category_length)
                server_response['Categories'][category_key].append(raw_data[text_index])
        text_index = text_index + 1    
    
    return server_response
    # pprint(server_response)
    
    
#Dictionary of Fairness and Category for mapping to server response

def FairnessKeys(server_key):
    keys = {0:'Arbitration',
            1:'Unilateral_Change',
            2:'Content_Removal',
            3:'Jurisdiction',
            4:'Choice_Of_Law',
            5:'Limitation_Of_Liability',
            6:'Unilateral_Termination',
            7:'Contract_By_Using'}
    return (keys.get(server_key))

def CategoryKeys(server_key):
    keys = {0:'DataRetention',
            1:'DataSecurity',
            2:'DoNotTrack',
            3:'FirstPartyCollection',
            4:'InternationalAndSpecificAudience',
            5:'Other',
            6:'PolicyChange',
            7:'ThirdPartySharingCollection',
            8:'UserAccessEditandDeletion',
            9:'UserChoiceControl' }
    return (keys.get(server_key))

#Dont Refer below this. Stuti directly performing preprocessing using keras. 
'''
def processing_data(data):
    filtered_msg = []
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(data)       #tokenization into words
    print(type(word_tokens))
    for word in word_tokens:
        if word not in stop_words:
            filtered_msg.append(word)
    filtered_msg = [stemmer.stem(word) for word in filtered_msg]
    #filtered_msg = [lemmatize.lemmatize(word,pos = "v") for word in filtered_msg]
    return filtered_msg
    #print(filtered_msg)
   
'''
# if __name__ == "__main__":
#     pre_process_driver(text)