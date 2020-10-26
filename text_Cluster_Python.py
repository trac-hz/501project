# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:58:08 2020

@author: Yiyang
"""

import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re   ## for regular expressions
from mpl_toolkits.mplot3d import Axes3D
#from nltk.stem.porter import PorterStemmer


# For text data
path="C:/Users/Yiyang/Desktop/501/assign/data/text"
#df=pd.read_csv(path,encoding="mbcs")

FileNameList=os.listdir(path)
print(FileNameList)

ListOfCompleteFilePaths=[]
for name in os.listdir(path):
    name=name.lower()
    next=path+"/"+name
    nextnameL=[re.findall(r'[a-z]+',name)[0]]
    nextname=nextnameL[0]
    ListOfCompleteFilePaths.append(next)
print(ListOfCompleteFilePaths)
#######################################################################
MyVectCount=CountVectorizer(input='filename',stop_words='english',max_features=100,encoding="mbcs")
MyVectTFIdf=TfidfVectorizer(input='filename',stop_words='english',max_features=100,encoding="mbcs")
DTM_Count=MyVectCount.fit_transform(ListOfCompleteFilePaths)
DTM_TF=MyVectTFIdf.fit_transform(ListOfCompleteFilePaths)
#############################################################
ColumnNames=MyVectCount.get_feature_names()
print("The vocab is: ",ColumnNames,"\n\n")

DF_Count=pd.DataFrame(DTM_Count.toarray(),columns=ColumnNames)
DF_TF=pd.DataFrame(DTM_TF.toarray(),columns=ColumnNames)
print(DF_Count)
print(DF_TF)

kmeans_object_Count=sklearn.cluster.KMeans(n_clusters=2)
kmeans_object_Count.fit(DF_Count)
labels=kmeans_object_Count.labels_
prediction_kmeans=kmeans_object_Count.predict(DF_Count)
print(prediction_kmeans)
Results=pd.DataFrame([DF_Count.index,labels]).T
print(Results)

x=DF_Count["movie"]
y=DF_Count["tv"]
colnames=DF_Count.columns
print(colnames)
fig1=plt.figure(figsize=(12,12))
ax1=Axes3D(fig1,rect=[0,0,0.9,1],elev=48,azim=134)
ax1.scatter(x,y,z,cmap="RdYlGn",edgecolor='k',s=200,c=prediction_kmeans)
ax1.w_xaxis.set_ticklabels([])
ax1.w_yaxis.set_ticklabels([])
ax1.w_zaxis.set_ticklabels([])
ax1.set_xlabel('Reading',fontsize=25)
ax1.set_ylabel('Students',fontsize=25)
ax1.set_zlabel('Books',fontsize=25)
centers=kmeans_object_Count.cluster_centers_
print(centers)
C1=centers[0,(1,2,14)]
print(C1)
C2=centers[1,(1,2,14)]
print(C2)
xs=C1[0],C2[0]
print(xs)
ys=C1[1],C2[1]
zs=C1[2],C2[2]
ax1.scatter(xs,ys,zs,c="black",s=2000,alpha=0.2)
plt.show()

mymatrix=DF_Count.values
cluster=AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")
p=cluster.fit_predict(mymatrix)
plt.figure(figsize=(10,7))
plt.scatter(Results.iloc[:,0],p,c=cluster.labels_,cmap="rainbow")
plt.title("k=2")
plt.show()

cluster=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
p=cluster.fit_predict(mymatrix)
plt.figure(figsize=(10,7))
plt.scatter(Results.iloc[:,0],p,c=cluster.labels_,cmap="rainbow")
plt.title("k=3")
plt.show()

cluster=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")
p=cluster.fit_predict(mymatrix)
plt.figure(figsize=(10,7))
plt.scatter(Results.iloc[:,0],p,c=cluster.labels_,cmap="rainbow")
plt.title("k=4")
plt.show()
######################################
Matrix=Results.values
dtm_euclidean=euclidean_distances(Matrix,Matrix)
euclidean_distances(Matrix,Matrix)
ax=sns.heatmap(dtm_euclidean)
plt.show()
##################################################
cosdist=1-cosine_similarity(DTM_TF)
print(np.round(cosdist,3))
linkage_matrix4=ward(cosdist)
names=FileNameList
fig,ax=plt.subplots(figsize=(15,20))
ax=dendrogram(linkage_matrix4,orientation="right",labels=names);
plt.tick_params(axis="x",which="both",bottom="on",top="on",labelbottom="on")
plt.tight_layout()
plt.show()

mandist=manhattan_distances(DTM_TF)
print(np.round(mandist,3))
linkage_matrix5=ward(mandist)
fig,ax=plt.subplots(figsize=(15,20))
ax=dendrogram(linkage_matrix5,orientation="right")
plt.tick_params(axis="x",which="both",bottom="on",top="on",labelbottom="on")
plt.tight_layout()
plt.show()

eucdist=euclidean_distances(DTM_TF)
print(np.round(eucdist,3))
linkage_matrix6=ward(eucdist)
fig,ax=plt.subplots(figsize=(15,20))
ax=dendrogram(linkage_matrix6,orientation="right")
plt.tick_params(axis="x",which="both",bottom="on",top="on",labelbottom="on")
plt.tight_layout()
plt.show()