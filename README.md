# Machine Learning Progect
Detect hate / offensive speech in greek tweets (homophobic/sexist).


# Overview
This project is an attemp to classify homophobic and sexist tweets in greek language, using Machine learning models.
In train data we have three classes of tweets, tweets with hate speech, tweets with offensive language and tweets without offensive or hate speech.
Unfortunately, the dataset isn't balanced.
The main problem with clafissification occurs beacause google translate not a quite efficient methon to translate slang language that did not follow the proper grammatic and syntax rules.

# Running
All the code and experiments for the project included in the classification.ipynb  jupyter notebook file. The code used for the translation included in tweet_translator.py.
# Classification Models
Four classification methods were used:
* Logistic Regression
* Guassian Naive Bayes
* KNearest Neighbors
* XGBoost
# Metrics
The metrics chosen for measuring each models performance are:
* Precision 
* Recall
* F1 score
* Accuracy
* Time
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

Logistic Regression time: 17.44202494621277 

              precision    recall  f1-score   support

           0       0.12      0.32      0.18       360
           1       0.85      0.66      0.74      4205
           2       0.34      0.46      0.39       874

    accuracy                           0.61      5439



GaussianNB run time: 1.778012752532959 

              precision    recall  f1-score   support

           0       0.06      0.58      0.12       360
           1       0.79      0.25      0.38      4205
           2       0.35      0.34      0.34       874

    accuracy                           0.29      5439

KNN run time: 107.33826065063477  

             precision    recall  f1-score   support

           0       0.12      0.18      0.14       360
           1       0.79      0.88      0.84      4205
           2       0.51      0.12      0.19       874

    accuracy                           0.71      5439



XGB run time: 211.92077612876892 
              
              precision    recall  f1-score   support

           0       0.35      0.03      0.05       360
           1       0.81      0.98      0.88      4205
           2       0.69      0.27      0.38       874

    accuracy                           0.80      5439


In the last step we evaluate our scores with k-fold.
Unfortunatey, it is clear now that none of this models consists a feasible solution to the problem.
# Final thoughts
The main problems occurs due to the fact that our dataset is inbalaned and our transation. 
In general sentiment lexicons and word embeddings constitute well-established sources of information for sentiment analysis in online social media. Although their effectiveness has been demonstrated in state-of-the-art sentiment analysis and related tasks in the English language, such publicly available resources are much less developed and evaluated for the Greek language.
# Installations
  pip install python-csv
  
  pip install pandas
  
  pip install scikit-learn
  
  pip istall xgboost
