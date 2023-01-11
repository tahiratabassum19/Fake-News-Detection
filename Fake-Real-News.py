import pandas as pd
# NLTK is our Natural-Language-Took-Kit
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# Libraries for helping us with strings
import string
# Regular Expression Library
import re
# text vectorizers: CountVectorizer && TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
 
# classifiers: MultinomialNB 
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
 
# Import some ML helper function
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
 
# Import our metrics to evaluate our model
from sklearn import metrics
from sklearn.metrics import classification_report
 
# Library for plotting
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')
 
import plotly.express as px
# Load Data
df = pd.read_csv('news_articles.csv')
#print(df.shape) 
#df.head(5)
# "Target" a column where Real = 1, Fake = 0
df['target'] = df.loc[:, 'label']
df = pd.get_dummies(df, columns=['target'], drop_first=True)
#keeping only english language
df = df[df.language == 'english']
# Inspect
df = df.dropna()
df.isnull().sum()
df = df.drop_duplicates()
#bring the root of the words
def root_words(string):
    porter = PorterStemmer()
   
    #  sentence into a list of words
    words = word_tokenize(string)
   
    valid_words = []
 
    for word in words:
       
        root_word = porter.stem(word)
       
        valid_words.append(root_word)
       
    string = ' '.join(valid_words)
 
    return string
 
def text_pipeline(input_string):
    input_string = root_words(input_string)
    return input_string
# if in dataset there were string without root words then can be used for later use
df['title_after'] = df['title_without_stopwords']
df['title_after'] = df['title_without_stopwords'].apply(text_pipeline)
#same process for text string cleaning 
df['text_after'] = df['text_without_stopwords']
df['text_after'] = df['text_without_stopwords'].apply(text_pipeline)
# TOP real title and text
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

#text
real_news2 = []
for x in df[df['target_Real'] == 1]['text_after'].tolist():
    for word in x.split():
        real_news2.append(word)
#text
pd.DataFrame(Counter(real_news2).most_common(10))
fake_news = []
for x in df[df['target_Real'] == 0]['title_after'].tolist():
    for word in x.split():
        fake_news.append(word)
#title
pd.DataFrame(Counter(fake_news).most_common(10))
#text
fake_news2 = []
for x in df[df['target_Real'] == 0]['text_after'].tolist():
    for word in x.split():
        fake_news2.append(word)
#text
pd.DataFrame(Counter(fake_news2).most_common(10))


# choosing feature for training data
X = df['title_after'].values +' ' +  df['type'].values +  ' ' + df['text_after'].values
y = df['label'].values
print(df['title_after'].values)

# Split our data into testing and training like always.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize our vectorizer
vectorizer = TfidfVectorizer()
# This makes vocab matrix
vectorizer.fit(X_train)
 
#This transforms documents into vectors.
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
features = vectorizer.get_feature_names()
weights = vectorizer.idf_
#print(len(features), len(weights))
df_idf = pd.DataFrame.from_dict( {'feature': features, 'idf': weights})
df_idf = df_idf.sort_values(by='idf', ascending=False)
df_idf
X_train[0]
 
# # Multinomial Naive Bayes
 
# Initalize our model.
mn_model = MultinomialNB(alpha=.05)
 
# Fit our model with our training data.
mn_model.fit(X_train, y_train)
 
# Make new predictions of our testing data.
y_pred = mn_model.predict(X_test)
 
 
# Make predicted probabilites of our testing data
y_pred_proba = mn_model.predict_proba(X_test)
 
# Evaluate our model
mn_accuracy =  mn_model.score(X_test, y_test)
 
# Print our evaluation metrics
print("Multinomial Naive bayes Model Accuracy: %f" % mn_accuracy)
#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred, target_names=mn_model.classes_))
 
 
# # Random Forest Classifier
# model takes longer to run

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
 
# Fit our model with our training data.
rf_model.fit(X_train, y_train)
# Make new predictions of our testing data.
y_pred = rf_model.predict(X_test)
 
 
# Make predicted probabilites of our testing data
y_pred_proba = rf_model.predict_proba(X_test)
 
# Evaluate our model
rf_accuracy =  rf_model.score(X_test, y_test)
 
# Print our evaluation metrics
#print("Random Forest Model Accuracy: %f" % rf_accuracy)
#print(classification_report(y_test, y_pred, target_names=rf_model.classes_))
 
# # # ADABoostClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=5)
# adab.fit(X_train,y_train)
# y_pred3 = adab.predict(X_test)
# ad_accuracy =  adab.score(X_test, y_test)
# #print("Ada Booster Model Accuracy: %f" % ad_accuracy)
# #print(classification_report(y_test, y_pred, target_names=adab.classes_))

import pickle
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(rf_model,open('rf_model.pkl','wb'))
pickle.dump(mn_model,open('mn.pkl','wb'))


# pickle.dump(adab,open('adab.pkl','wb'))
 
 

