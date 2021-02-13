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
* Recall
* F1 score
* Accuracy
# Data
For this project, the dataset that was used may be found in the bellow github project:
https://github.com/t-davidson/hate-speech-and-offensive-language, and translate it with tweet_transator.py .

# Process
The experiment can be divide into five steps:
1. Data translation 
2. Text analysis
3. Feature extraction
4. Model tuning
5. Model evaluation

In firts step, we transalte our dataset.
Second step in about reprocess in our tweets using stopwords, find spaces, urls, retweets, tokensize.
In third step we use Tfidf vectorize, pos tags and sentiment analysis in order to extract our final features.
In fourth step we run our classification model and find for all classifiers run time and classification scores:
Logistic Regression time: 13.659311294555664 

              precision    recall  f1-score   support
           0       0.13      0.36      0.19       135
           1       0.84      0.58      0.69      1384
           2       0.32      0.51      0.39       294
    accuracy                           0.55      1813




GaussianNB run time: 0.8815405368804932 

              precision    recall  f1-score   support
           0       0.08      0.64      0.14       135
           1       0.79      0.23      0.36      1384
           2       0.33      0.34      0.33       294
    accuracy                           0.28      1813


KNN run time: 32.43400049209595 

              precision    recall  f1-score   support
           0       0.16      0.23      0.19       135
           1       0.79      0.88      0.83      1384
           2       0.51      0.10      0.17       294

    accuracy                           0.71      1813



XGB run time: 168.84296131134033 

              precision    recall  f1-score   support
           0       0.50      0.01      0.03       135
           1       0.79      0.98      0.87      1384
           2       0.64      0.20      0.31       294

    accuracy                           0.78      1813


In the last step we evaluate our scores with k-fold.
Unfortunatey, it is clear now that none of this models consists a feasible solution to the problem.
