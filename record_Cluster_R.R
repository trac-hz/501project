library(factoextra)
library(dbscan)
library(tm)
library(textstem)
library(stats)
library(NbClust)
library(cluster)
library(mclust)
library(amap)
library(purrr)
library(stylo)
library(philentropy)
library(SnowballC)
library(caTools)
library(dplyr)
library(stringr)

# For Record Data
setwd("C:/Users/Yiyang/Desktop/501/assign/data")
Record<-read.csv("movies_revised.csv")
Record<-Record[c(1:100),]
head(Record)
str(Record)
Label<-Record$decades
Record<-Record[,-c(5)]
head(Record)
## Create a normalized version of Spotify_DF
Record_Norm<-as.data.frame(apply(Record[,1:4],2,
                                 function(x) (x-min(x))/(max(x)-min(x))))
### Look at the pairwise distances between the vectors 
Dist_norm<-dist(Record_Norm,method="minkowski",p=2)

fviz_nbclust(Record,kmeans,method="wss")
fviz_nbclust(Record,kmeans,method="silhouette")
fviz_nbclust(Record,kmeans,method="gap_stat")
#######################################################
distance_M<-dist(Record,method="manhattan")
distance_E<-dist(Record,method="euclidean")
distance_C<-dist(Record,method="canberra")

groups_M<-hclust(distance_M,method="ward.D")
plot(groups_M,cex=0.4,hang=0.2)
rect.hclust(groups_M,k=2)

groups_E<-hclust(distance_E,method="ward.D")
plot(groups_E,cex=0.4,hang=0.2)
rect.hclust(groups_E,k=2)

groups_C<-hclust(distance_C,method="ward.D")
plot(groups_C,cex=0.4,hang=0.2)
rect.hclust(groups_C,k=2)
#####################################################
km<-kmeans(Record,2,nstart=30)
fviz_cluster(km,data=Record,geom="point",ellipse=FALSE,
             show.clust.cent=FALSE,palette="jco",
             ggtheme=theme_classic())

kmeans<-NbClust::NbClust(Record_Norm,min.nc=2,max.nc=5,method="kmeans")
barplot(table(kmeans$Best.n[1,]),
        xlab="Numer of Clusters",ylab="",
        main="Number of Clusters")

kmeans_Result2<-kmeans(Record,2,nstart=25) 
fviz_cluster(kmeans_Result2,Record,main="k=2")
kmeans_Result3<-kmeans(Record,3,nstart=25) 
fviz_cluster(kmeans_Result3,Record,main="k=3")
kmeans_Result4<-kmeans(Record,4,nstart=25) 
fviz_cluster(kmeans_Result4,Record,main="k=4")

db_record<-dbscan(as.matrix(Record_Norm),0.5,2)
hullplot(as.matrix(Record_Norm),db_record$cluster)

# For Text Data
SmallCorpus<-Corpus(DirSource("Corpus"))
(getTransformations())
(ndocs<-length(SmallCorpus))
SmallCorpus<-tm_map(SmallCorpus,content_transformer(tolower))
SmallCorpus<-tm_map(SmallCorpus,removePunctuation)
SmallCorpus<-tm_map(SmallCorpus,removeWords,stopwords("english"))
SmallCorpus<-tm_map(SmallCorpus,lemmatize_strings)
SmallCorpus_DTM<-DocumentTermMatrix(SmallCorpus,
                                    control=list(
                                      stopwords=TRUE,
                                      wordLengths=c(3,10),
                                      removePunctuation=TRUE,
                                      removeNumbers=TRUE,
                                      tolower=TRUE))
inspect(SmallCorpus_DTM)
SmallCorpus_TERM_DM<-TermDocumentMatrix(SmallCorpus,
                                        control=list(
                                          stopwords=TRUE,
                                          wordLengths=c(3,10),
                                          removePunctuation=TRUE,
                                          removeNumbers=TRUE,
                                          tolower=TRUE))

inspect(SmallCorpus_TERM_DM)
SmallCorpus_DF_DT<-as.data.frame(as.matrix(SmallCorpus_DTM))
SmallCorpus_DF_TermDoc<-as.data.frame(as.matrix(SmallCorpus_TERM_DM))
SC_DTM_mat<-as.matrix(SmallCorpus_DTM)
(SC_DTM_mat[1:12,1:10])
SC_TERM_Doc_mat<-as.matrix(SmallCorpus_TERM_DM)
(SC_TERM_Doc_mat[1:12,1:10])

(SmallCorpusWordFreq<-colSums(SC_DTM_mat))
(head(SmallCorpusWordFreq))
(length(SmallCorpusWordFreq))
ord<-order(SmallCorpusWordFreq)
(SmallCorpusWordFreq[head(ord)])
(SmallCorpusWordFreq[tail(ord)])
(Row_Sum_Per_doc<-rowSums((SC_DTM_mat)))
SC_DTM_mat_norm<-apply(SC_DTM_mat,1,function(i) round(i/sum(i),2))
(SC_DTM_mat_norm[1:12,1:5])

fviz_nbclust(SmallCorpus_DF_DT,method="silhouette", 
             FUN=hcut,k.max=9)
fviz_nbclust(SmallCorpus_DF_DT,method="gap_stat", 
             FUN=hcut,k.max=9)
fviz_nbclust(SmallCorpus_DF_DT,method="wss", 
             FUN=hcut,k.max=9)

SC_DTM_mat_norm_t<-t(SC_DTM_mat_norm)
kmeans_smallcorp_Result<-kmeans(SC_DTM_mat_norm_t,4,nstart=25)   
print(kmeans_smallcorp_Result)
kmeans_smallcorp_Result$centers
cbind(SmallCorpus_DF_DT,cluster=kmeans_smallcorp_Result$cluster)
kmeans_smallcorp_Result$cluster
kmeans_smallcorp_Result$size
fviz_cluster(kmeans_smallcorp_Result,SmallCorpus_DF_DT, 
             main="k=4",repel=TRUE)
kmeans_smallcorp_Result2<-kmeans(SC_DTM_mat_norm_t,2,nstart=25)
fviz_cluster(kmeans_smallcorp_Result2,SmallCorpus_DF_DT, 
             main="k=2",repel=TRUE)
kmeans_smallcorp_Result3<-kmeans(SC_DTM_mat_norm_t,3,nstart=25)
fviz_cluster(kmeans_smallcorp_Result3,SmallCorpus_DF_DT, 
             main="k=3",repel=TRUE)

distance_M<-dist(SmallCorpus_DF_DT,method="manhattan")
distance_E<-dist(SmallCorpus_DF_DT,method="euclidean")
distance_C<-dist(SmallCorpus_DF_DT,method="canberra")
groups_M<-hclust(distance_M,method="ward.D")
plot(groups_M,cex=0.4,hang=-1)
rect.hclust(groups_M,k=3)
groups_E<-hclust(distance_E,method="ward.D")
plot(groups_E,cex=0.4,hang=-1)
rect.hclust(groups_E,k=3)
groups_C<-hclust(distance_C,method="ward.D")
plot(groups_C,cex=0.4,hang=-1)
rect.hclust(groups_C,k=3)

db_text<-dbscan(as.matrix(SmallCorpus_DF_DT),2.5,3)
hullplot(as.matrix(SmallCorpus_DF_DT),db_text$cluster)