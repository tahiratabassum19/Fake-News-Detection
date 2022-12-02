#Import Streamlit 
import streamlit as st

# Import pandas for data handling
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


# classifiers: MultinomialNB && RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


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

# You may need to download these from nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')

#others
import plotly.express as px

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://media.istockphoto.com/id/1278709873/photo/brown-recycled-paper-crumpled-texture-background-cream-old-vintage-page-or-grunge-vignette.jpg?b=1&s=170667a&w=0&k=20&c=NqKmm_gkRwJAqpTbiiqv3TwfWjq9ymOwUDwfG2ck9no=");
             #background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# ## "Target" a column where Real = 1, Fake = 0

df['target'] = df.loc[:, 'label']
df = pd.get_dummies(df, columns=['target'], drop_first=True) # why do we need drop first as its two type only ?
df.head(2)

df.language.value_counts()

#keeping only english language
df = df[df.language == 'english']

df.language.value_counts()

df.info()

#  Inspect 
print(df.isnull().sum())

df = df.dropna()


print(df.shape)

df.isnull().sum()
print(df.duplicated().sum())

df = df.drop_duplicates()

print(df.shape, 'after')

print(df.duplicated().sum())

#  Find Label balances.
df.label.value_counts()

#  Find Type balances.
df.type.value_counts()

print("Original TEXT:", df['title'][10],":::REAL or FAKE:", df['label'][10])

print("Clean TEXT from data:", df['title_without_stopwords'][10],":::REAL or FAKE:", df['label'][10])
print("ORIGINAL TEXT:", df['text'][90])

print("Clean TEXT from data:", df['text_without_stopwords'][90])

colors = ['#63666A','#d6cfc7']
plt.pie(df['label'].value_counts(), labels=['Fake','Real'],autopct="%0.2f", colors = colors)
plt.show()

print(df.label.value_counts())
sns.countplot(x=df["label"])

# colors = ['#808588','#d6cfc7']
print(df.type.value_counts())
plt.figure(figsize=(19,5))
sns.countplot(x= 'type', data= df, color='#332570', saturation = 0.1)

#bs = all are fake news
plt.figure(figsize=(20,10))
sns.countplot(x= 'type', hue= 'target_Real', data= df)

fig = px.sunburst(df, path=['label', 'type'])
fig.show()

### Stop words are already removed
### Punctuation are already removed
### Already in Lowercase

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

sent = 'I played and started playing with players and we all love to play with plays'
root_words(sent)

def text_pipeline(input_string):
    input_string = root_words(input_string)
    #input_string=remove_punc(input_string) # to remove punctuatuion

#     input_string= get_top_n_words(input_string)
    return input_string


#function to remove punctuation from user entered sentence
def remove_punc(punc_string):
    punc_string=re.sub(r'[^\w\s]',' ', punc_string)
    return punc_string 
    
a='hel??lo gi**rl'
remove_punc(a)
'''
# might not need this part for the df as it already comes wth pipelining 
df['title_after'] = df['title_without_stopwords']
df['title_after'] = df['title_without_stopwords'].apply(text_pipeline)
print("Clean TEXT from data:", df['title_without_stopwords'][0])
print("CLEANDED TEXT:", df['title_after'][0])
# might not need this too
df['text_after'] = df['text_without_stopwords']
df['text_after'] = df['text_without_stopwords'].apply(text_pipeline)
'''
## TOP real title and text
from collections import Counter

#title
#not sure 
real_news = []
for x in df[df['target_Real'] == 1]['title_after'].tolist():
    for word in x.split():
        real_news.append(word)

#title
real_news = []
for x in df[df['target_Real'] == 1]['title'].tolist():
    for word in x.split():
        real_news.append(word)

len(real_news)

# sns.barplot(pd.DataFrame(Counter(real_news).most_common(30))[0],pd.DataFrame(Counter(real_news).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()

from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n = None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    freq_sorted = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return freq_sorted[:n]


top_unigram = get_top_n_words(df['text_without_stopwords'], 10)
words = [i[0] for i in top_unigram]
count = [i[1] for i in top_unigram]

plt.figure(figsize=(10,7))
plt.bar(words, count,align='center')
plt.xticks(rotation=90)
plt.ylabel('Number of Occurences')
plt.show()

#title
df2 = pd.DataFrame(Counter(real_news).most_common(10))
df2

# df2.groupby('word').sum()['count'].sort_values(ascending=False)
# fig=px.bar(df2,x='word',y='count',color='count',title='Top 10 bigrams')
# fig.show()

#title
#removed after title_after with title only 
fake_news = []
for x in df[df['target_Real'] == 0]['title'].tolist():
    for word in x.split():
        fake_news.append(word)
#title
pd.DataFrame(Counter(fake_news).most_common(10))

# Split the data into testing and training
X = df['title'].values +' ' +  df['type'].values +  ' ' + df['text'].values 

y = df['label'].values
print(df['title'].values)


# Split our data into testing and training like always. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Save the raw text for later just incase
X_train_text = X_train
X_test_text = X_test


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize our vectorizer
vectorizer = TfidfVectorizer()

# This makes your vocab matrix
vectorizer.fit(X_train)

# This transforms your documents into vectors.
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape, type(X))


features = vectorizer.get_feature_names()
weights = vectorizer.idf_

print(len(features), len(weights))

df_idf = pd.DataFrame.from_dict( {'feature': features, 'idf': weights})

df_idf = df_idf.sort_values(by='idf', ascending=False)

df_idf


X_train[0]

# Multinomial Naive Bayes

# Initalize our model.
model = MultinomialNB(alpha=.05)

# Fit our model with our training data.
model.fit(X_train, y_train)

# Make new predictions of our testing data. 
y_pred = model.predict(X_test)


# Make predicted probabilites of our testing data
y_pred_proba = model.predict_proba(X_test)

# Evaluate our model
accuracy =  model.score(X_test, y_test)

# Print our evaluation metrics
print("Model Accuracy: %f" % accuracy)


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=model.classes_))


#random forest classifier 

from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier()


# Fit our model with our training data.
rf_model.fit(X_train, y_train)


# Make new predictions of our testing data. 
y_pred = rf_model.predict(X_test)


# Make predicted probabilites of our testing data
y_pred_proba = rf_model.predict_proba(X_test)

# Evaluate our model
accuracy =  rf_model.score(X_test, y_test)

# Print our evaluation metrics
print("Model Accuracy: %f" % accuracy)

print(classification_report(y_test, y_pred, target_names=rf_model.classes_))

#bernoulli b

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)

accuracy =  bnb.score(X_test, y_test)
print("Model Accuracy: %f" % accuracy)
print(classification_report(y_test, y_pred, target_names=bnb.classes_))


# ADABoostClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=5)
adab.fit(X_train,y_train)
y_pred3 = adab.predict(X_test)

accuracy =  adab.score(X_test, y_test)
print("Model Accuracy: %f" % accuracy)
print(classification_report(y_test, y_pred, target_names=adab.classes_))


from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred)

cm = cm.round(2)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(cm, annot=True, cmap='YlOrBr', fmt='g')
plt.title("Confusion Matrix of Real or Fake")
plt.xlabel('Predicted')
plt.ylabel('Actual')

print('true-negitive:', tn, 
      '\nfalse-positive:', fp, 
      '\nfalse-negative:', fn, 
      '\ntrue-positive:', tp )


new_text = 'NEWS RELEASE: DOH CITES SIX COMPANIES FOR AIR PERMIT VIOLATIONS'

new_text = text_pipeline(new_text)


print(new_text)

new_text_vectorized = vectorizer.transform([new_text])

print("Your News is:", model.predict(new_text_vectorized))

# streamlit starts here 
import streamlit as st

st.set_page_config(page_title="Fake News Detection", page_icon=":tada", layout="wide")
st.subheader("We are detecting fake articles")
st.write("we are passionate about identifying fake news ")


