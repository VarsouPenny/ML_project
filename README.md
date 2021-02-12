# Machine Learning Progect
Detect hate / offensive speech in greek tweets (homophobic/sexist).


# Overview
This project is an attemp to classify homophobic and sexist tweets in greek language, using Machine learning models.
In train data we have three classes of tweets, tweets with hate speech, tweets with offensive language and tweets without offensive or hate speech.
Unfortunately, the dataset isn't balanced.
The main problem with clafissification occurs beacause google translate isnt quit efficient methon to translate slang language that did not follow the proper grammatic and syntax rules.

# Running
All the cide and experiments for the project included in the classification.ipynb  jupyter notebook file. The code used for the translation included in tweet_translator.py.
# Classification Models
Four classification methods were used:
* Logistic Regression
* Guassin Naive Bayes
* KNearest Neighbors
* XGBoost
# Metrics
The metrics chosen for measuring each models performance are:
* Precision
* F1 score
* Recall
# Data
For this project, the dataset that was used may be found in the bellow github project:
https://github.com/t-davidson/hate-speech-and-offensive-language, and translate it with tweet_transator.py .
