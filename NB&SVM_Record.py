# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:18:27 2020

@author: Yiyang
"""
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os

#from nltk.stem import WordNetLemmatizer 
#LEMMER = WordNetLemmatizer() 

from nltk.stem.porter import PorterStemmer
STEMMER=PorterStemmer()

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words

import string
import numpy as np
from sklearn.model_selection import train_test_split
import random as rd
rd.seed(1234)

## Read the data into a dataframe
filename="C:/Users/Yiyang/Desktop/501/assign/data/movies.csv"
FinalDF=pd.read_csv(filename)
print(FinalDF)


FinalDF=FinalDF.loc[:,["budget","gross","runtime","score"]]
df1=FinalDF[FinalDF.score>=6]
df2=FinalDF[FinalDF.score<6]
df1["score_cat"]="high"
df2["score_cat"]="lOW"
FinalDF = df1.append(df2)

## clean the data
import random
items = list(range(1,6820))
random.shuffle(items)   

FinalDF=FinalDF.loc[items,["budget","gross","runtime","score_cat"]]
print(FinalDF)

TrainDF, TestDF = train_test_split(FinalDF, test_size=0.3)


### TEST ---------------------
TestLabels=TestDF["score_cat"]
print(TestLabels)

## remove labels
TestDF = TestDF.drop(["score_cat"], axis=1)
print(TestDF)

## TRAIN ----------------------------
TrainLabels=TrainDF["score_cat"]
print(TrainLabels)

## remove labels
TrainDF = TrainDF.drop(["score_cat"], axis=1)
print(TrainDF)

####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB1= MultinomialNB()

## When you look up this model, you learn that it wants the 

## Run on all three Dfs.................
MyModelNB1.fit(TrainDF, TrainLabels)
Prediction1 = MyModelNB1.predict(TestDF)

print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(TestLabels)



## confusion matrix
from sklearn.metrics import confusion_matrix


cnf_matrix1 = confusion_matrix(TestLabels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)


print(np.round(MyModelNB1.predict_proba(TestDF),2))

import seaborn as sns
sns.heatmap(cnf_matrix1, square=True, annot=True, fmt='d')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')



#######################################################
### Bernoulli #########################################
#######################################################


#############################################
###########  SVM ############################
#############################################
from sklearn.svm import LinearSVC
SVM_Model=LinearSVC(C=10)
SVM_Model.fit(TrainDF, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model.predict(TestDF))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

## plot
sns.heatmap(SVM_matrix, square=True, annot=True, fmt='d')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


#############################################
###########  SVM ############################
#############################################


TRAIN= TrainDF
TRAIN_Labels= TrainLabels
TEST= TestDF
TEST_Labels= TestLabels


SVM_Model1=LinearSVC(C=50)
SVM_Model1.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model1.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

## plot
sns.heatmap(SVM_matrix, square=True, annot=True, fmt='d')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


#############################################
########### other kernels and change the cost
#############################################

#--------------
## RBF
SVM_Model2=sklearn.svm.SVC(C=1, kernel='rbf', 
                           verbose=True, gamma="auto")
SVM_Model2.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model2.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

## plot
sns.heatmap(SVM_matrix, square=True, annot=True, fmt='d')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


