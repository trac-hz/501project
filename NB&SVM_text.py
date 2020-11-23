# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:40:41 2020

@author: Yiyang
"""
## Textmining Naive Bayes Example
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
import random as rd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.svm import LinearSVC
##--------------------- importing data-------------------
text = pd.read_csv('C:/Users/Yiyang/Desktop/501/assign/data/IMDB Dataset_label.csv',encoding="ANSI")

text1 = text[text.sentiment == 'positive']
text2 = text[text.sentiment == 'negative']
###--------------------Wordcloud-------------------------
stopwords = nltk.corpus.stopwords.words('english')
def word_cloud(textfile):
    comment_string = '' 
    
    for val in textfile.review: 
        val = str(val) 
        tokens = val.split('|') 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
            tokens[i] = tokens[i]+','
        comment_string += " ".join(tokens)+","
    
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_string) 
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
      
    plt.show() 
word_cloud(text1)
word_cloud(text2)

###----------------Put text data by level into seperate folders------
def bagging(var,name):
    file = 'C:/Users/Yiyang/Desktop/501/assign/data/corpus/'+name+'/Document{}.txt'
    for i, row in var.iterrows():
        with open(file.format(i), 'w', encoding="utf-8") as f:
            f.write(str(row['review']))
            
bagging(text1, 'positive')
bagging(text2,'negative')

###--------------- Getting into the folder:-------------------

STEMMER=PorterStemmer()
## Create my own stemmer
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words

analyzer=CountVectorizer().build_analyzer()

MyVect_STEM = CountVectorizer(input='filename', # take out stopwords, symbols and numbers
                        stop_words=stopwords,
                        #max_features=500,
                        analyzer='word',
                        token_pattern=r'\b[a-zA-Z]{2,}\b',
                        encoding="ISO-8859-1",
                        tokenizer=MY_STEMMER)


MyVect_STEM_Bern=CountVectorizer(input='filename', # take out stopwords, symbols and numbers
                        stop_words=stopwords,
                        #max_features=500,
                        analyzer='word',
                        token_pattern=r'\b[a-zA-Z]{2,}\b',
                        encoding="ISO-8859-1",
                        tokenizer=MY_STEMMER,
                        binary=True
                        )

### ---------------------vectorzing-----------------------
FinalDF=pd.DataFrame()
for name in ["positive","negative"]:
    path="C:/Users/Yiyang/Desktop/501/assign/data/corpus/"+name
    FileList=[]
    for item in os.listdir(path):
        next=path+ "\\" + item
        FileList.append(next)  
        print("full list...")
    X5= MyVect_STEM.fit_transform(FileList)
    ColumnNames2=MyVect_STEM.get_feature_names()
    builder=pd.DataFrame(X5.toarray(),columns=ColumnNames2)
    builder["Label"]=name
    
    FinalDF = FinalDF.append(builder)

FinalDF=FinalDF.fillna(0)
#print(FinalDF)
def RemoveNums(SomeDF):
    #print(SomeDF)
    print("Running Remove Numbers function....\n")
    temp=SomeDF
    MyList=[]
    for col in temp.columns:
        #print(col)
        #Logical1=col.isdigit()  ## is a num
        Logical2=str.isalpha(col) ## this checks for anything
        ## that is not a letter
        if(Logical2==False):# or Logical2==True):
            #print(col)
            MyList.append(str(col))
            #print(MyList)       
    temp.drop(MyList, axis=1, inplace=True)
            #print(temp)
            #return temp
       
    return temp

FinalDF = RemoveNums(FinalDF)
FinalDF.to_csv('FinalDF.csv')
## ----------------------Splitting---------------------------
rd.seed(78)
TrainDF, TestDF = train_test_split(FinalDF, test_size=0.3)


## Drop labels from test set
TestLabels=TestDF["Label"]
TestDF = TestDF.drop(["Label"], axis=1)

#####-------------------Naive Bayes--------------------
MyModelNB= MultinomialNB()
TrainDF_nolabels=TrainDF.drop(["Label"], axis=1)
TrainLabels=TrainDF["Label"]
model = MyModelNB.fit(TrainDF_nolabels, TrainLabels)

# Compare predictions and actual labels
Prediction = MyModelNB.predict(TestDF)
cnf_matrix = confusion_matrix(TestLabels, Prediction)


### visualization

sns.heatmap(cnf_matrix.T, square=True, annot=True, fmt='d',
            xticklabels=["pos","neg"], 
            yticklabels=["pos","neg"])
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
print(sklearn.metrics. classification_report(TestLabels, Prediction))

#####################    SVM    #####################

SVM_Model=LinearSVC(C=10)
SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model.predict(TestDF))
print(sklearn.metrics.classification_report(TestLabels, SVM_Model.predict(TestDF)))
################# RBF model#######################
SVM_Model2=sklearn.svm.SVC(C=100, kernel='rbf', 
                           verbose=True, gamma="auto")

SVM_Model2.fit(TrainDF_nolabels, TrainLabels)

SVM_matrix2 = confusion_matrix(TestLabels, SVM_Model2.predict(TestDF))
print(sklearn.metrics.classification_report(TestLabels, SVM_Model2.predict(TestDF)))
#################Poly model##########################
SVM_Model3=sklearn.svm.SVC(C=100, kernel='poly',degree=2,
                           gamma="auto", verbose=True)

SVM_Model3.fit(TrainDF_nolabels, TrainLabels)

SVM_matrix3 = confusion_matrix(TestLabels, SVM_Model3.predict(TestDF))
print(sklearn.metrics.classification_report(TestLabels, SVM_Model3.predict(TestDF)))

## visualize Confusion Matrix1
sns.heatmap(SVM_matrix.T, square=True, annot=True, fmt='d', 
            xticklabels=["pos","neg"], 
            yticklabels=["pos","neg"])
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

## visualize Confusion Matrix2
sns.heatmap(SVM_matrix2.T, square=True, annot=True, fmt='d', 
            xticklabels=["pos","neg"], 
            yticklabels=["pos","neg"])
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

## visualize Confusion Matrix3
sns.heatmap(SVM_matrix3.T, square=True, annot=True, fmt='d', 
            xticklabels=["pos","neg"], 
            yticklabels=["pos","neg"])
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
