# -*- coding: utf-8 -*-
"""
Created on Fri May 12 00:30:29 2023

@author: yasin
"""

import pandas as pd

# %%
data = pd.read_csv(r'data/gender_classifier.csv',encoding='latin1')
data = pd.concat([data.gender,data.description],axis=1) #data'yı iki koluma ayırdık
data.dropna(axis=0,inplace=True) #boş satırları atıyoruz
data.gender = [1 if i == 'female' else 0 for i in data.gender]

# %% cleaning data
# regular expression RE

import re

first_description = data.description[4]
description = re.sub('[^a-zA-z]',' ',first_description) #^ işareti olmayanları boşlukla değiştir demek (a-zA-Z olmayanları)
description = description.lower() #tüm harfleri büyük harf yapar

# %% stopwords (irrelavent words) alakasz kelimeler
import nltk #natural language tool kit
nltk.download('stopwords')   #corpus diye bir klasöre indiriliyor
nltk.download('punkt')
from nltk.corpus import stopwords  # corpus klasöründen import ediyoruz

# description = description.split()

# split yerine tokenizer kullanabiliriz
description = nltk.word_tokenize(description)

#split kullanarak shouldn't ı should ve not olarak ikiye ayıramayız.
# %%
#gereksiz kelimeleri çıkar
description = [word for word in description if not word in set(stopwords.words('english'))]

# %% lemmataziton (kelimenin köklerini bulma)

import nltk as nlp
nlp.download('wordnet')
nlp.download('omw-1.4')

# %% köklerini bul
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
# %% listeyi boşluklarla bir birleştir
description = ' '.join(description)

# %% Data Cleaning

description_list = []

for description in data.description:
    description = re.sub('[^a-zA-z]',' ',description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [word for word in description if not word in set(stopwords.words('english'))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = ' '.join(description)
    description_list.append(description)
    
# %% bag of words

from sklearn.feature_extraction.text import CountVectorizer #bag of words yaratmak için kullanacağız
max_features = 500 #en çok kullanılan kelime sayısı. İstersek arttırabiliriz

count_vectorizer = CountVectorizer(max_features=max_features, stop_words='english') #CountVectorizer ile stopwords, tokenize,lower işlemleri yapılaiblir

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() #x

print('en sık kullanılan {} kelimeler: {}'.format(max_features,count_vectorizer.get_feature_names()))

# %%

y = data.iloc[:,0].values #male or female classes
x = sparce_matrix

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

# %%

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#prediction
y_pred =nb.predict(x_test)
print('accuracy: ',nb.score(x_test,y_test))

















