# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 12:23:07 2021

@author: EngineerErdem
"""

import pandas as pd
import numpy as np 
import re # regullar expression
import nltk #nltk import library for stop words key
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#sendimental polarity
stopping=nltk.download("stopwords")
ps=PorterStemmer() # separate stem
comments = pd.read_csv('Restaurant_Reviews.csv', error_bad_lines=False)
# data preprocessing
comment=[]
for i in range(716):
    client=re.sub("[^a-zA-Z]"," ",comments["Review"][i]) #CLEAR PUNCUATİON POİNTS
    client=client.lower() # we are made smaller
    client=client.split() # we have a list becasue split function.
    client=[ps.stem(word) for word in client if not word in set(stopwords.words("english"))]
    client=" ".join(client) # capsulation
    comment.append(client)
# end preproccesing and starting feature 
from sklearn.feature_extraction.text import CountVectorizer
count_vec=CountVectorizer(max_features=2700)
X=count_vec.fit_transform(comment).toarray()
y=comments.iloc[:,1].values.astype(int)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =10)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

print(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

