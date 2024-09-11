import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string
import LogicalRegression as LR_file


#Creating a function to process the texts
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


data_path = '' #Inserire il proprio path dei CSV

df_fake = pd.read_csv(data_path+'Fake.csv')
df_true = pd.read_csv(data_path+'True.csv')

#print(df_fake.head())


df_fake['class'] = 0
df_true['class'] = 1

#print(df_fake.shape), print(df_true.shape)

# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i],axis=0,inplace=True)

df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i],axis=0,inplace=True)

#print(df_fake.shape), print(df_true.shape)

#df_fake_manual_testing["class"] = 0
#df_true_manual_testing["class"] = 1

#print(df_fake_manual_testing.head(10))

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing],axis=0)
df_manual_testing.to_csv(data_path+'manual_testing.csv')


#Merging True and Fake Dataframes

df_merge = pd.concat([df_fake,df_true],axis=0)
#print(df_merge.head(10))
#print(df_merge.columns)

#Removing columns which are not required
df = df_merge.drop(['title','subject','date'], axis=1)
#print(df.isnull().sum())

#Random Shuffling the dataframe
df = df.sample(frac=1)
#print(df.head())
df.reset_index(inplace=True)
df.drop(['index'],axis=1, inplace=True)
#print(df.columns)
#print(df.head())

df["text"] = df["text"].apply(wordopt)

#Defining dependent and independent variables
x = df['text']
y = df['class']

#Splitting Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
#print(y)


#Convert text to vectors
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


#Logistic Regression
LR_file.Start(xv_train, y_train, xv_test, y_test, 1) #modificare ultimo parametro a 0 per non produrre report


#news = str(input())
#manual_testing(news)