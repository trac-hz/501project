# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:51:29 2020

@author: Yiyang
"""
## Module 5

import nltk
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sklearn
import re  
from nltk.corpus import stopwords
import random as rd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.stem.porter import PorterStemmer
#import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from nltk.stem.porter import PorterStemmer

import random
import pandas as pd
import matplotlib.pyplot as plt


###-----------------------create tags -------------------------

df = pd.read_csv('C:/Users/Yiyang/Desktop/501/assign/data/IMDB Dataset_label.csv',encoding="mbcs")
df1 = df.sample(frac=0.01, replace=True, random_state=1)

AllReviewsList=df1.iloc[:,0].tolist() 
AllLabelsList=df1.iloc[:,1].tolist() 

########################################
##
## CountVectorizer  and TfidfVectorizer
##
########################################
## Now we have what we need!
## We have a list of the contents (reviews)
## in the csv file.

STEMMER=PorterStemmer()
print(STEMMER.stem("fishings"))

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words

My_CV1=CountVectorizer(input='content',
                       analyzer = 'word',
                        stop_words='english',
                        max_features=100               
                        )



My_CV1_Bern=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        max_features=100, 
                        tokenizer=MY_STEMMER,
                        lowercase = True,
                        binary=True
                        )


My_TF1=TfidfVectorizer(input='content',
                       analyzer = 'word',
                        stop_words='english',
                        max_features=100                    
                        )


My_TF1_STEM=TfidfVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        max_features=100,  
                        tokenizer=MY_STEMMER,
                        lowercase = True,
                        )
#
## NOw I can vectorize using my list of complete paths to my files
X_CV1=My_CV1.fit_transform(AllReviewsList)
X_TF1=My_TF1.fit_transform(AllReviewsList)

X_CV1_Bern=My_CV1.fit_transform(AllReviewsList)
X_TF1_STEM=My_CV1.fit_transform(AllReviewsList)

print(My_CV1.vocabulary_)
print(My_TF1.vocabulary_)

#print(X_CV1_Bern.vocabulary_)
#print(X_TF1_STEM.vocabulary_)

## Let's get the feature names which ARE the words
## The column names are the same for TF and CV
ColNames=My_TF1.get_feature_names()

## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CV=pd.DataFrame(X_CV1.toarray(), columns=ColNames)
DataFrame_TF=pd.DataFrame(X_TF1.toarray(), columns=ColNames)

DataFrame_CV_Bern=pd.DataFrame(X_CV1_Bern.toarray(), columns=ColNames)
DataFrame_TF_STEM=pd.DataFrame(X_TF1_STEM.toarray(), columns=ColNames)

## Drop/remove columns not wanted
print(DataFrame_CV.columns)
## Let's build a small function that will find 
## numbers/digits and return True if so
##------------------------------------------------------
### DEFINE A FUNCTION that returns True if numbers
##  are in a string 
def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------
for nextcol in DataFrame_CV.columns:
    #print(nextcol)
    ## Remove unwanted columns
    #Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    ##-------------call the function -------
    LogResult=Logical_Numbers_Present(nextcol)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)
        DataFrame_CV_Bern=DataFrame_CV_Bern.drop([nextcol], axis=1)
        DataFrame_TF_STEM=DataFrame_TF_STEM.drop([nextcol], axis=1)

    ## The following will remove any column with name
    ## of 3 or smaller - like "it" or "of" or "pre".
    ## print(len(nextcol))  ## check it first
    ## NOTE: You can also use this code to CONTROL
    ## the words in the columns. For example - you can
    ## have only words between lengths 5 and 9. 
    ## In this case, we remove columns with words <= 3.
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)
        DataFrame_CV_Bern=DataFrame_CV_Bern.drop([nextcol], axis=1)
        DataFrame_TF_STEM=DataFrame_TF_STEM.drop([nextcol], axis=1)        
    
print(DataFrame_CV)
print(DataFrame_TF)
####################################################
##
## Adding the labels to the data......
## This would be good if you are modeling with
## supervised methods - such as NB, SVM, DT, etc.
##################################################
## Recall:
print(AllLabelsList)
print(type(AllLabelsList))

## Place these on dataframes:
## List --> DF
DataFrame_CV.insert(loc=0, column='Label', value=AllLabelsList)
DataFrame_TF.insert(loc=0, column='Label', value=AllLabelsList)
DataFrame_CV_Bern.insert(loc=0, column='Label', value=AllLabelsList)
DataFrame_TF_STEM.insert(loc=0, column='Label', value=AllLabelsList)

#print(DataFrame_CV)
#print(DataFrame_TF)

############################################
##
##  WRITE CLEAN, Tokenized, vectorized data
##  to new csv file. This way,  you can read it
##  into any program and work with it.
##
######################################################
CV_File="MyTextOutfile_count.csv"
TF_File="MyTextOutfile_Tfidf.csv"
CV_Bern_File="MyTextOutfile_count.csv"
TF_STEM_File="MyTextOutfile_Tfidf.csv"

################ Save csv directly --
DataFrame_CV.to_csv(CV_File, index = False)
DataFrame_TF.to_csv(TF_File, index = False)
DataFrame_CV_Bern.to_csv(CV_File, index = False)
DataFrame_TF_STEM.to_csv(TF_File, index = False)

##################################################
##
##        Now we have 4 labeled dataframes!
##        Let's model them.....
##
######################################################

## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.
from sklearn.model_selection import train_test_split
import random as rd
rd.seed(1234)
TrainDF1, TestDF1 = train_test_split(DataFrame_CV, test_size=0.3)
TrainDF2, TestDF2 = train_test_split(DataFrame_TF, test_size=0.3)
TrainDF3, TestDF3 = train_test_split(DataFrame_CV_Bern, test_size=0.3)
TrainDF4, TestDF4 = train_test_split(DataFrame_TF_STEM, test_size=0.3)


###############################################
## For all three DFs - separate LABELS
#################################################
## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
### TEST ---------------------
Test1Labels=TestDF1["Label"]
Test2Labels=TestDF2["Label"]
Test3Labels=TestDF3["Label"]
Test4Labels=TestDF4["Label"]
print(Test2Labels)

## remove labels
TestDF1 = TestDF1.drop(["Label"], axis=1)
TestDF2 = TestDF2.drop(["Label"], axis=1)
TestDF3 = TestDF3.drop(["Label"], axis=1)
TestDF4 = TestDF4.drop(["Label"], axis=1)
print(TestDF1)

## TRAIN ----------------------------
Train1Labels=TrainDF1["Label"]
Train2Labels=TrainDF2["Label"]
Train3Labels=TrainDF3["Label"]
Train4Labels=TrainDF4["Label"]
print(Train3Labels)

## remove labels
TrainDF1 = TrainDF1.drop(["Label"], axis=1)
TrainDF2 = TrainDF2.drop(["Label"], axis=1)
TrainDF3 = TrainDF3.drop(["Label"], axis=1)
TrainDF4 = TrainDF4.drop(["Label"], axis=1)
print(TrainDF3)

####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()

## Run on all three Dfs.................
NB1=MyModelNB.fit(TrainDF1, Train1Labels)
Prediction1 = MyModelNB.predict(TestDF1)
print(np.round(MyModelNB.predict_proba(TestDF1),2))

NB2=MyModelNB.fit(TrainDF2, Train2Labels)
Prediction2 = MyModelNB.predict(TestDF2)
print(np.round(MyModelNB.predict_proba(TestDF2),2))

NB3=MyModelNB.fit(TrainDF3, Train3Labels)
Prediction3 = MyModelNB.predict(TestDF3)
print(np.round(MyModelNB.predict_proba(TestDF3),2))

NB4=MyModelNB.fit(TrainDF4, Train4Labels)
Prediction4 = MyModelNB.predict(TestDF4)
print(np.round(MyModelNB.predict_proba(TestDF4),2))

print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)

print("\nThe prediction from NB is:")
print(Prediction3)
print("\nThe actual labels are:")
print(Test3Labels)

print("\nThe prediction from NB is:")
print(Prediction4)
print("\nThe actual labels are:")
print(Test4Labels)

######################### confusion matrix ############################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

## The confusion matrix is square and is labels X labels
## We have two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
cnf_matrix1 = confusion_matrix(Test1Labels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)


cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
print("\nThe confusion matrix is:")
print(cnf_matrix2)

cnf_matrix3 = confusion_matrix(Test3Labels, Prediction3)
print("\nThe confusion matrix is:")
print(cnf_matrix3)

cnf_matrix4 = confusion_matrix(Test4Labels, Prediction4)
print("\nThe confusion matrix is:")
print(cnf_matrix4)


#########################################################
#############    Decision Trees   #######################
#########################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix

#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
MyDT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

## ------------------------------
## This for loop will fit and predict Decision Trees for 
## all 4 of the dataframes. Notice that this uses dynamic variables
## and eval
##--------------------------
#print(TrainDF1.columns)
for i in [1]:
    temp1=str("TrainDF"+str(i))
    temp2=str("Train"+str(i)+"Labels")
    temp3=str("TestDF"+str(i))
    temp4=str("Test"+str(i)+"Labels")
    
    ## perform DT
    MyDT.fit(eval(temp1), eval(temp2))
    
    ## plot the tree
    tree.plot_tree(MyDT)
    plt.savefig(temp1)
    feature_names=eval(str(temp1+".columns"))
    dot_data = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=eval(str(temp1+".columns")),  
                      #class_names=MyDT.class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)                                    
    graph = graphviz.Source(dot_data) 
    
    ## Create dynamic graph name
    #tempname=str("Graph" + str(i))
    #graph.render(tempname) 
    
    ## Show the predictions from the DT on the test set
    print("\nActual for DataFrame: ", i, "\n")
    print(eval(temp2))
    print("Prediction\n")
    DT_pred=MyDT.predict(eval(temp3))
    print(DT_pred)
    
    ## Show the confusion matrix
    bn_matrix = confusion_matrix(eval(temp4), DT_pred)
    print("\nThe confusion matrix is:")
    print(bn_matrix)
    FeatureImp=MyDT.feature_importances_   
    indices = np.argsort(FeatureImp)[::-1]
    
    ## print out the important features.....
    for f in range(TrainDF4.shape[1]):
        if FeatureImp[indices[f]] > 0:
            print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
            print ("feature name: ", feature_names[indices[f]])

## FYI for small datasets you can zip features....
## print(dict(zip(iris_pd.columns, clf.feature_importances_)))


#########################################################
##
##                 Random Forest for Text Data
##
#################################################################
RF = RandomForestClassifier()
RF.fit(TrainDF1, Train1Labels)
RF_pred=RF.predict(TestDF1)

bn_matrix_RF_text = confusion_matrix(Test1Labels, RF_pred)
print("\nThe confusion matrix is:")
print(bn_matrix_RF_text)

################# VIS RF---------------------------------
## FEATURE NAMES...................
FeaturesT=TrainDF1.columns
#Targets=StudentTestLabels_Num

figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

tree.plot_tree(RF.estimators_[0],
               feature_names = FeaturesT, 
               #class_names=Targets,
               filled = True)

##save it
figT.savefig('RF_Tree_Text')  ## creates png


#####------------------> View estimator Trees in RF

figT2, axesT2 = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)

for index in range(0, 5):
    tree.plot_tree(RF.estimators_[index],
                   feature_names = FeaturesT, 
                   filled = True,
                   ax = axesT2[index])

    axesT2[index].set_title('Estimator: ' + str(index), fontsize = 11)
## Save it
figT2.savefig('FIVEtrees_RF.png')


#################-------------------------->
## Feature importance in RF
##-----------------------------------------
## Recall that FeaturesT are the columns names - the words in this case.
######
FeatureImpRF=RF.feature_importances_   
indicesRF = np.argsort(FeatureImpRF)[::-1]
## print out the important features.....
for f2 in range(TrainDF1.shape[1]):   ##TrainDF1.shape[1] is number of columns
    if FeatureImpRF[indicesRF[f2]] >= 0.01:
        print("%d. feature %d (%.2f)" % (f2 + 1, indicesRF[f2], FeatureImpRF[indicesRF[f2]]))
        print ("feature name: ", FeaturesT[indicesRF[f2]])
        

## PLOT THE TOP 10 FEATURES...........................
top_ten_arg = indicesRF[:10]
#print(top_ten_arg)
plt.title('Feature Importances Positive and Negative')
plt.barh(range(len(top_ten_arg)), FeatureImpRF[top_ten_arg], color='b', align='center')
plt.yticks(range(len(top_ten_arg)), [FeaturesT[i] for i in top_ten_arg])
plt.xlabel('Relative Importance')
plt.show()






