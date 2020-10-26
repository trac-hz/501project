# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:13:13 2020

@author: Yiyang
"""

import os
import re
import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# For Record Data
data="C:/Users/Yiyang/Desktop/501/assign/data/movies_revised.csv"
df=pd.read_csv(data,encoding='unicode_escape')
print(df.head())
TrueLabel=df["decades"]
print(TrueLabel)
df=df.drop(["decades"],axis=1)
print(df.head())

kmeans_object=sklearn.cluster.KMeans(n_clusters=4)
kmeans=kmeans_object.fit(df)
prediction_kmeans=kmeans_object.predict(df)
print(prediction_kmeans)

data_classes=["80s","90s","00s","10s"]

dc=dict(zip(data_classes,range(0,4)))
print(dc)
TrueLabel_num=TrueLabel.map(dc,na_action="ignore")
print(TrueLabel_num)

fig2=plt.figure(figsize=(12,12))
ax2=Axes3D(fig2,rect=[0,0,0.9,1],elev=48,azim=134)
print(df)
x=df.iloc[:,0]
y=df.iloc[:,1]
z=df.iloc[:,2]
print(x,y,z)
ax2.scatter(x,y,z,cmap="RdYlGn",edgecolor="k",s=200,c=prediction_kmeans)
ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel("Computing", fontsize=25)
ax2.set_ylabel("Sport", fontsize=25)
ax2.set_zlabel("Medical", fontsize=25)
plt.show()
centers=kmeans.cluster_centers_
print(centers)
print(centers[0,0])
xs=(centers[0,0],centers[1,0])
ys=(centers[0,1],centers[1,1])
zs=(centers[0,2],centers[1,2])
ax2.scatter(xs,ys,zs,c="black",s=2000,alpha=0.2)
plt.show()

sns.set(font_scale=3)
x=df.values
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
df_scaled=pd.DataFrame(x_scaled)
sns.heatmap(df_scaled)

kmeans_record=KMeans(n_clusters=2).fit(df_scaled)
centroids=kmeans_record.cluster_centers_
plt.scatter(df_scaled.iloc[:,0],df_scaled.iloc[:,1],c=kmeans_record.labels_)
plt.scatter(centroids[:,0],centroids[:,1],c="red")
plt.title("k=2")
plt.show()

kmeans_record=KMeans(n_clusters=3).fit(df_scaled)
centroids=kmeans_record.cluster_centers_
plt.scatter(df_scaled.iloc[:,0],df_scaled.iloc[:,1],c=kmeans_record.labels_)
plt.scatter(centroids[:,0],centroids[:,1],c="red")
plt.title("k=3")
plt.show()

kmeans_record=KMeans(n_clusters=4).fit(df_scaled)
centroids=kmeans_record.cluster_centers_
plt.scatter(df_scaled.iloc[:,0],df_scaled.iloc[:,1],c=kmeans_record.labels_)
plt.scatter(centroids[:,0],centroids[:,1],c="red")
plt.title("k=4")
plt.show()

cosdist=1-cosine_similarity(df)
print(np.round(cosdist,3))
linkage_matrix1=ward(cosdist)
fig,ax=plt.subplots(figsize=(15,20))
ax=dendrogram(linkage_matrix1,orientation="right")
plt.tick_params(axis="x",which="both",bottom="on",top="on",labelbottom="on")
plt.tight_layout()
plt.show()

mandist=manhattan_distances(df)
print(np.round(mandist,3))
linkage_matrix2=ward(mandist)
fig,ax=plt.subplots(figsize=(15,20))
ax=dendrogram(linkage_matrix2,orientation="right")
plt.tick_params(axis="x",which="both",bottom="on",top="on",labelbottom="on")
plt.tight_layout()
plt.show()

eucdist=euclidean_distances(df)
print(np.round(eucdist,3))
linkage_matrix3=ward(eucdist)
fig,ax=plt.subplots(figsize=(15,20))
ax=dendrogram(linkage_matrix3,orientation="right")
plt.tick_params(axis="x",which="both",bottom="on",top="on",labelbottom="on")
plt.tight_layout()
plt.show()