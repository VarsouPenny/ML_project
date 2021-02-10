#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importa pandas library for inmporting CSV
import pandas as pd 

# Imports the Google Cloud client library
from google_trans_new import google_translator  

# Instantiates a client
translate_client = google_translator()  


#Translating the text to specified target language
def translate(word):
    # Target language
    target_language = 'el' #Add here the target language that you want to translate to
    # Translates some text into Russian
    translation = translate_client.translate(
        word,
        lang_tgt=target_language)

    return (translation)

#Import data from CSV
def importCSV():
    data = pd.read_csv('labeled_data.csv',encoding='latin1')
    countRows = (len(data))
    #Create a dictionary with translated words
    translatedCSV = { "id":[] , "count":[],"hate_speech":[], "offensive_language":[],"neither":[],"class":[],"tweet":[]} #Change the column names accordingly to your coumns names
 
    #Translated word one by one from the CSV file and save them to the dictionary
    for index, row in data.iterrows():
#         translated_tweet = translate(row[0])
        translatedCSV["id"].append(row[0])
        translatedCSV["count"].append(row[1])
        translatedCSV["hate_speech"].append(row[2])
        translatedCSV["offensive_language"].append(row[3])
        translatedCSV["neither"].append(row[4])
        translatedCSV["class"].append(row[5])
        translated_tweet = translate(row[6])
        translatedCSV["tweet"].append(translated_tweet)

               #        translatedCSV["Text"].append(translate(row["Text"]))
#         translatedCSV["column2"].append(translate(row["column2"]))

    #Create a Dataframe from Dictionary 
    #Save the DataFrame to a CSV file
    df = pd.DataFrame(data=translatedCSV)
    df.to_csv("mytrain_dataset.csv", sep='\t',index=False)
    


# In[2]:


importCSV()

