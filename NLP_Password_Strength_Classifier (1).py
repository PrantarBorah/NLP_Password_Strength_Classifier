#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train=pd.read_csv(r"C:\Users\Manjit.Borah\Downloads/Data.csv",error_bad_lines= False)


# In[3]:


train.head()


# In[4]:


train['strength'].unique()


# In[5]:


train.isnull().sum()


# In[6]:


train.dropna(inplace=True)


# In[7]:


train.isnull().sum()


# In[8]:


sns.countplot(train['strength'])


# In[9]:


password_tuple= np.array(train)
password_tuple


# In[10]:


import random
random.shuffle(password_tuple)


# In[11]:


x=[labels[0]for labels in password_tuple]
y=[labels[1]for labels in password_tuple]
x


# In[13]:


#TF-IDF value matrix analysis (TF*IDF is the final matrix that is to be passed to the ML Algorithm)


# In[17]:


def word_to_char(input):
    character=[]
    for i in input:
        character.append(i)
    return character


# In[18]:


word_to_char('184520socram')


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[24]:


vectorizer=TfidfVectorizer(tokenizer=word_to_char)


# In[27]:


X=vectorizer.fit_transform(x)


# In[29]:


X.shape


# In[30]:


vectorizer.get_feature_names()


# In[31]:


first_doc_vect=X[0]
first_doc_vect


# In[32]:


first_doc_vect.T.todense()


# In[34]:


df=pd.DataFrame(first_doc_vect.T.todense(),index=vectorizer.get_feature_names(),columns=['TF*IDF_values'])
df.sort_values(by=['TF*IDF_values'],ascending =False)


# In[35]:


#Logistic Regression


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3)


# In[40]:


X_train.shape


# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


clf=LogisticRegression(random_state=0,multi_class='multinomial')


# In[43]:


clf.fit(X_train,Y_train)


# In[44]:


Y_pred=clf.predict(X_test)
Y_pred


# In[45]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[46]:


cm=confusion_matrix(Y_test,Y_pred)


# In[47]:


print(cm)


# In[48]:


print(accuracy_score(Y_test,Y_pred))


# In[49]:


from sklearn.metrics import classification_report


# In[50]:


print(classification_report(Y_test,Y_pred))


# In[ ]:




