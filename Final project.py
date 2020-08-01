#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)


# In[21]:


tweets_df = pd.read_csv('tweets.csv')


# In[22]:


tweets_df


# In[23]:


tweets_df.info()


# In[24]:


tweets_df.describe()


# In[25]:


tweets_df['tweet']


# In[26]:


tweets_df = tweets_df.drop(['id'],axis = 1)


# In[27]:


tweets_df


# In[28]:


sns.countplot(tweets_df['label'], label = 'Count')


# In[29]:


tweets_df['length'] = tweets_df['tweet'].apply(len)


# In[30]:


tweets_df


# In[31]:


tweets_df['length'].plot(bins=100, kind='hist')


# In[32]:


tweets_df.describe()


# In[33]:


positive = tweets_df[tweets_df['label']==0]


# In[34]:


positive


# In[35]:


negative = tweets_df[tweets_df['label']==1]


# In[36]:


negative


# In[37]:


sentences = tweets_df['tweet'].tolist()


# In[38]:


sentences


# In[39]:


len(sentences)


# In[40]:


sentences_as_one_string = " ".join(sentences)


# In[41]:


get_ipython().system('pip install WordCloud')
from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[42]:


negative_list = negative['tweet'].tolist()
negative_sentences_as_one_string = " ".join(negative_list)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))


# In[43]:


import string
string.punctuation


# In[44]:


Test = 'Good morning beautiful people :)... I am having fun learning ML and AI!!'


# In[45]:


Test_punc_removed = [char for char in Test if char not in string.punctuation ]


# In[46]:


Test_punc_removed


# In[47]:


Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[48]:


import nltk

nltk.download('stopwords')


# In[49]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This paper is the second paper.','And this is the thrid paper.']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[51]:


print(vectorizer.get_feature_names())


# In[52]:


print(X.toarray())


# In[53]:


def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[54]:


tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)


# In[55]:


print(tweets_df_clean[5])


# In[56]:


print(tweets_df['tweet'][5])


# In[57]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = message_cleaning)
tweets_countvectorizer = CountVectorizer(analyzer = message_cleaning, dtype = 'uint8').fit_transform(tweets_df['tweet'])


# In[58]:


tweets_countvectorizer.shape


# In[59]:


X = tweets_countvectorizer


# In[60]:


y = tweets_df['label']


# In[61]:


X.shape


# In[62]:


y.shape


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[64]:


from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[65]:


from sklearn.metrics import classification_report, confusion_matrix


# In[66]:


y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)


# In[67]:


print(classification_report(y_test, y_predict_test))


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg_classifier = LogisticRegression()
logreg_classifier.fit(X_train, y_train)


# In[69]:


from sklearn.metrics import classification_report, confusion_matrix


# In[71]:


print(classification_report(y_test, y_predict_test))






