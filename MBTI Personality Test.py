#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing useful libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize


# In[2]:


#importing warnings to avoid it from coming into outputs

import warnings
warnings.filterwarnings('ignore')


# In[5]:


path="C:\\Users\\IQRA\\Desktop\\Projects\\mbti_1.csv"
df=pd.read_csv(path)


# In[6]:


print(df)


# In[7]:


#train and test split

train_data=df.sample(frac=0.75)
test_data=df.drop(train_data.index)


# In[8]:


print(f"No. of training examples: {train_data.shape[0]}")
print(f"No. of testing examples: {test_data.shape[0]}")


# In[9]:


#plotting graph for number of personality types

f, ax = plt.subplots(figsize=(10, 10))
sns.countplot(train_data['type'].sort_values(ascending=True))
plt.title("Count of Personality Types")
plt.xlabel("Personality Type")
plt.ylabel("Counts")


# In[10]:


#converting personality types into encoding attributes

train_data['E/I'] = train_data['type'].apply(lambda x: x[0] == 'E').astype('int')
train_data['S/N'] = train_data['type'].apply(lambda x: x[1] == 'N').astype('int')
train_data['T/F'] = train_data['type'].apply(lambda x: x[2] == 'T').astype('int')
train_data['J/P'] = train_data['type'].apply(lambda x: x[3] == 'J').astype('int')
train_data.head()


# In[11]:


y_train = train_data[['E/I', 'S/N', 'T/F', 'J/P']]

#merging test and train datasets into one dataframe (along with new additions made)

merged_data = pd.concat((train_data, test_data)).reset_index(drop=True)


# In[12]:


print("merged_data size is : {}".format(merged_data.shape))


# In[13]:


#replacing the ||| in between with , 

merged_data['split_posts'] = merged_data['posts'].str.split('\|\|\|')
merged_data['split_posts'] = merged_data['split_posts'].apply(', '.join)


# In[14]:


#transforming all the text to lower case

merged_data['split_posts'] = merged_data['split_posts'].str.lower()


# In[15]:


#transforming urls into string objects

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
merged_data['split_posts'] = merged_data['split_posts'].replace(to_replace=pattern_url, value=subs_url, regex=True)


# In[16]:


#remiving punctuations

def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])
merged_data['posts_no_punct'] = merged_data['split_posts'].apply(remove_punctuation)


# In[17]:


#transforming sentences into individual words

merged_data['words'] = merged_data['posts_no_punct'].apply(word_tokenize)


# In[18]:


merged_data.head(10)


# In[19]:


ntrain = train_data.shape[0]
ntest = test_data.shape[0]

train_wordclouds = merged_data[:ntrain]


# In[20]:


#grouping the data by personality type for wordcloud

group_wordclouds = train_wordclouds[['type','words']]
group_wordclouds = group_wordclouds.groupby('type').sum()
group_wordclouds = group_wordclouds.reset_index()


# In[21]:


group_wordclouds.head()


# In[22]:


fig, ax = plt.subplots(nrows=4, ncols=4)
fig.set_size_inches(22, 10)

random = group_wordclouds['words']
for i, j in group_wordclouds.iterrows():
    text = ', '.join(random[i])
    
    wordcloud = WordCloud().generate(text)
    plt.subplot(4, 4, (i+1))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(str(group_wordclouds['type'].iloc[i]))


# In[23]:


group_wordclouds = group_wordclouds['words']

vocab = []
for i in random:
    vocab.append(i)

flat_vocab = []
for sublist in vocab:
    for item in sublist:
        flat_vocab.append(item)

text = ', '.join(word for word in flat_vocab)


# In[24]:


wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('All Vocabulary Wordcloud')
plt.show() 


# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[26]:


remove_stopwords = TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.5, min_df=0.01, max_features=10000)
merged_data_TFIDF = remove_stopwords.fit_transform(merged_data['posts'])


# In[27]:


merged_data_TFIDF.shape


# In[28]:


train = merged_data_TFIDF[:ntrain]
test = merged_data_TFIDF[ntrain:]
print("Train dataset size is : {} ".format(train.shape))
print("Test dataset size is : {} ".format(test.shape))
print("Y-Train dataset size is : {} ".format(y_train.shape))


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import optuna


# In[30]:


def log_loss_cv(model, category):
    log_loss = -cross_val_score(model, train, y_train[category], scoring="neg_log_loss", cv=5)
    return(log_loss)


# In[31]:


logreg_EI = make_pipeline(LogisticRegression())
logreg_SN = make_pipeline(LogisticRegression())
logreg_TF = make_pipeline(LogisticRegression())
logreg_JP = make_pipeline(LogisticRegression())


# In[32]:


EI_score = log_loss_cv(logreg_EI, 'E/I')
SN_score = log_loss_cv(logreg_SN, 'S/N')
TF_score = log_loss_cv(logreg_TF, 'T/F')
JP_score = log_loss_cv(logreg_JP, 'J/P')


# In[33]:


print('Extrovert(E)/Introvert(I) Score: ', EI_score.mean())
print('Sensing(S)/Intuition(N) Score: ', SN_score.mean())
print('Thinking(T)/Feeling(F) Score: ', TF_score.mean())
print('Judging(J)/Percieving(P) Score: ', JP_score.mean()) 


# In[66]:


def objective(trial):
     penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
     tol = trial.suggest_loguniform('tol', 1e-10, 1)
     C = trial.suggest_loguniform('C', 1e-10, 1)
     random_state = trial.suggest_int('random_state', 1, 10)
     max_iter = trial.suggest_int('max_iter', 1000, 10000)
     warm_start = trial.suggest_categorical('warm_start', [True, False])
    
     classifier_obj = LogisticRegression(penalty=penalty, 
                                         tol=tol, 
                                         C=C, 
                                         random_state=random_state, 
                                         max_iter=max_iter, 
                                         warm_start=warm_start)
     x = train
     y = y_train['E/I']

     score = cross_val_score(classifier_obj, x, y, scoring="neg_log_loss")
     accuracy = score.mean()
        
     return 1.0 - accuracy


# In[67]:


study = optuna.create_study()
study.optimize(objective, n_trials=1000)


# In[68]:


study.best_params


# In[76]:


logreg_EI = make_pipeline(LogisticRegression(penalty='l2',
                                             tol= 2.6321107832201454e-07,
                                             C= 0.9729565980304679,
                                             random_state= 7,
                                             max_iter= 8157,
                                             warm_start= False))

logreg_SN = make_pipeline(LogisticRegression(penalty='l2',
                                             tol= 2.6321107832201454e-07,
                                             C= 0.9729565980304679,
                                             random_state= 7,
                                             max_iter= 8157,
                                             warm_start= False))

logreg_TF = make_pipeline(LogisticRegression(penalty='l2',
                                             tol= 2.6321107832201454e-07,
                                             C= 0.9729565980304679,
                                             random_state= 7,
                                             max_iter= 8157,
                                             warm_start= False))

logreg_JP = make_pipeline(LogisticRegression(penalty='l2',
                                             tol= 2.6321107832201454e-07,
                                             C= 0.9729565980304679,
                                             random_state= 7,
                                             max_iter= 8157,
                                             warm_start= False))


# In[77]:


EI_score = log_loss_cv(logreg_EI, 'E/I')
SN_score = log_loss_cv(logreg_SN, 'S/N')
TF_score = log_loss_cv(logreg_TF, 'T/F')
JP_score = log_loss_cv(logreg_JP, 'J/P')


# In[80]:


print('Extrovert/Introvert Score: ', EI_score.mean())
print('Sensing/Intuition Score: ', SN_score.mean())
print('Thinking/Feeling Score: ', TF_score.mean())
print('Judging/Percieving Score: ' ,JP_score.mean())


# In[81]:


logreg_EI.fit(train, y_train['E/I'])
logreg_SN.fit(train, y_train['S/N'])
logreg_TF.fit(train, y_train['T/F'])
logreg_JP.fit(train, y_train['J/P'])


# In[89]:


plt.figure(figsize=(8,8));
x = [13.8, 12.3, 11.6, 8.8, 8.7, 8.5, 8.1, 5.4, 4.4, 4.3, 3.3, 3.2, 2.5, 2.1, 1.8, 1.5]
labels = ['ISFJ', 'ESFJ', 'ISTJ', 'ISFP', 'ESTJ', 'ESFP', 'ENFP', 'ISTP', 'INFP', 'ESTP', 'INTP', 'ENTP', 'ENFJ', 'INTJ', 'ENTJ', 'INFJ']
plt.pie(x, labels=labels, autopct='%1.1f%%')
plt.show()


# In[ ]:




