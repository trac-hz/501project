library(ggpubr)
## create test and train datasets:
df <- read.csv("C:/Users/Yiyang/Desktop/501/assign/data/movies.csv")
# shuffle the dataset
set.seed(9850) 
u_num <- runif(nrow(df))
Newdf <- df[order(u_num),]
(summary(Newdf))

df1<-df[c(1:1000),]
#delete column company, director, name, star, writer
df1<-df1[,-c(1,2,4,7,11,12,13,14)]
head(df1)
quantile(df1$score)
### categorize variables score:
df1$score_cat <- 
  cut(df1$score, breaks=c(-Inf, 6.0, Inf), 
      labels=c("Low_score","High_score"))
df1<-df1[,-c(6)]
### categorize variables country:
df1$country[df1$country == "Argentina"] <- 'South America'
df1$country[df1$country == "Australia"] <- 'Oceania'
df1$country[df1$country == "Belgium"] <- 'Europe'
df1$country[df1$country == "Canada"] <- 'North America'
df1$country[df1$country == "Denmark"] <- 'Europe'
df1$country[df1$country == "France"] <- 'Europe'
df1$country[df1$country == "Hong Kong"] <- 'Asia'
df1$country[df1$country == "Hungary"] <- 'Europe'
df1$country[df1$country == "Iran"] <- 'Asia'
df1$country[df1$country == "Ireland"] <- 'Europe'
df1$country[df1$country == "Israel"] <- 'Asia'
df1$country[df1$country == "Italy"] <- 'Europe'
df1$country[df1$country == "Japan"] <- 'Asia'
df1$country[df1$country == "Netherlands"] <- 'Europe'
df1$country[df1$country == "Spain"] <- 'Europe'
df1$country[df1$country == "Sweden"] <- 'Europe'
df1$country[df1$country == "Switzerland"] <- 'Europe'
df1$country[df1$country == "UK"] <- 'Europe'
df1$country[df1$country == "USA"] <- 'North America'
df1$country[df1$country == "West Germany"] <- 'Europe'

sum(is.na(df1)) 
##as factor
df1$country<-as.factor(df1$country)
df1$genre<-as.factor(df1$genre)
df1$rating<-as.factor(df1$rating)

## ---------------------Now we can split the data--------------------:
n <- round(nrow(df1)/5)
s <- sample(1:nrow(df1), n)

Test <- df1[s,]
Train <- df1[-s,]
## Record Data----------------------------
#################################################
#NB
#remove label for test data
Test_NoLabel<- Test[,-c(6)]

(NB_movie<-naiveBayes(score_cat~., data=Train))
NB_Pred_movie <- predict(NB_movie, Test_NoLabel)
(cm<-table(NB_Pred_movie,Test$score_cat))
plot(Test$score_cat, main = 'Actual label')
plot(NB_Pred_movie, main = 'Predicted label')

(conf <- confusionMatrix(data = NB_Pred_movie, reference = Test$score_cat))
################################SVM#################
# SVM
SVM_L <- svm(score_cat~., data=Train, 
             kernel='linear', cost=.1, 
             scale=FALSE)
#SVM_P <- svm(score_cat~., data=Train, 
#             kernel='polynomial', cost=.1, 
#             scale=FALSE)
SVM_R <- svm(score_cat~., data=Train, 
             kernel='radial', cost=.1, 
             scale=FALSE)
print(SVM_L)
##Prediction 
(pred_L <- predict(SVM_L, Test_NoLabel, type="class"))
(pred_R <- predict(SVM_R, Test_NoLabel, type="class"))
## COnfusion Matrix
(Ptable_L <- table(pred_L, Test$score_cat))
(Ptable_R <- table(pred_R, Test$score_cat))
## Misclassification Rate for Polynomial
(MR_L <- 1 - sum(diag(Ptable_L))/sum(Ptable_L))
(MR_R <- 1 - sum(diag(Ptable_R))/sum(Ptable_R))
## We have 5 variables and so need our plot to be more precise
plot(SVM_L, data=Train, gross~runtime)
plot(SVM_R, data=Train, gross~runtime)




## text Data----------------------------
#######################################################
positive <- Corpus(DirSource("C:\\Users\\Yiyang\\Desktop\\501\\assign\\data\\corpus\\positive"))
negative <- Corpus(DirSource("C:\\Users\\Yiyang\\Desktop\\501\\assign\\data\\corpus\\negative"))
# Get vectorized data
positive_dtm <- DocumentTermMatrix(positive,
                                 control = list(
                                   wordLengths=c(4, 8), stopwords = TRUE,
                                   removePunctuation = TRUE,
                                   removeNumbers = TRUE,
                                   tolower=TRUE,
                                   remove_separators = TRUE,
                                   stemming = TRUE))
negative_dtm <- DocumentTermMatrix(negative,
                                control = list(
                                  wordLengths=c(4, 8),stopwords = TRUE,
                                  removePunctuation = TRUE,
                                  removeNumbers = TRUE,
                                  tolower=TRUE,
                                  remove_separators = TRUE,
                                  stemming = TRUE))
# Convert DocumentTerm to matrix
positive_matrix <- as.matrix(positive_dtm)
negative_matrix <- as.matrix(negative_dtm)

# Make wordclouds for each label:
##positive:
v1 <- sort(rowSums(t(positive_matrix)),decreasing=TRUE)
d1 <- data.frame(word = names(v1),freq=v1)

set.seed(124)
wordcloud(words = d1$word, freq = d1$freq, min.freq = 1,
          max.words=2000, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

barplot(d1[1:10,]$freq, las = 2, names.arg = d1[1:10,]$word,
        main ="Most frequent words in posotive reviews",
        ylab = "Word frequencies")

##negative:
v2 <- sort(rowSums(t(negative_matrix)),decreasing=TRUE)
d2 <- data.frame(word = names(v2),freq=v2)

set.seed(14)
wordcloud(words = d2$word, freq = d2$freq, min.freq = 1,
          max.words=2000, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

barplot(d2[1:10,]$freq, las = 2, names.arg = d2[1:10,]$word,
        main ="Most frequent words in negative reviews",
        ylab = "Word frequencies")

# Split data into train and test sets
df_pos <- as.data.frame(positive_matrix)
df_neg <- as.data.frame(negative_matrix)
df_pos$Label <- 'positive'
df_neg$Label <- 'negative'

## combine three datasets with different labels together
df_mix <- rbind.fill(df_pos,df_neg)
df_mix[is.na(df_mix)] <- 0
#sum(is.na(df_mix))

set.seed(55)
X = 3   ## This will create a 1/3, 2/3 split. 
(s<-seq(1,nrow(df_mix),X)) # take 1/3 data from df_mix

## Use these X indices to make the Testing and then
## Training sets:

TestSet<-df_mix[s, ] ##len = 223
TrainSet<-df_mix[-s, ] ## len = 444
str(TestSet$Label)

## convert type of Label to factor
TestSet$Label <- as.factor(TestSet$Label)
TrainSet$Label <- as.factor(TrainSet$Label)

## remove Label from Test data and Train data
Test_label <- TestSet$Label
Test_num <- TestSet[ , -which(names(TestSet) == 'Label')]

Train_label <- TrainSet$Label
Train_num <- TrainSet[ , -which(names(TrainSet) == 'Label')]

## store it to csv
write.csv(TestSet, "C:\\Users\\Yiyang\\Desktop\\501\\assign\\data\\NBTestSet_text.csv")
write.csv(TrainSet, "C:\\Users\\Yiyang\\Desktop\\501\\assign\\data\\NBTrainSet_text.csv")

##############################################################################
##################################### NB #####################################
##############################################################################
NB<-naiveBayes(Train_num, Train_label, laplace = 1)
NB_pred <- predict(NB, Test_num)
NB
(tb2 <- table(NB_pred,Test_label))
(n <- 1 - sum(diag(tb2))/sum(tb2))
##Visualize
plot(Test_label, col = c('lightblue','pink','purple'), main = 'Actual label')
plot(NB_pred, col = c('lightblue','pink','purple'), main = 'Predicted label')

################################SVM#################
# SVM
SVM_L <- svm(Label~., data=TrainSet, 
             kernel='linear', cost=.1, 
             scale=FALSE)
print(SVM_L)
pred_L <- predict(SVM_L, Test_num, type="class")

(Ltable <- table(pred_L, Test_label))
##
SVM_R <- svm(Label~., data=TrainSet, 
             kernel='radial', cost=.1, 
             scale=FALSE)
print(SVM_R)

pred_R <- predict(SVM_R, Test_num, type="class")

(Rtable <- table(pred_R, Test_label))

(MR_R <- 1 - sum(diag(Rtable))/sum(Rtable))
(MR_L <- 1 - sum(diag(Ltable))/sum(Ltable))
##Linear plot:
plot(Test_label,main="true label")
plot(pred_L,main="predicted label for linear SVM")
plot(pred_R,main="predicted label for Polynomial SVM")
## Feature imp
IMP <- function (model, top_features){
  w <- t(model$coefs) %*% model$SV               # weight vectors
  w <- apply(w, 2, function(v){sqrt(sum(v^2))})  # weight
  w <- sort(w, decreasing = T)
  #head(w,10)
  df_w <- as.data.frame(w)
  ranking <- rownames(df_w)
  tf <- ranking[1:top_features]
  table <- data.frame(tf,df_w$w[1:10])
  colnames(table) <- c('top features','weight')
  return(table)
}
SVM_L_plot <- IMP(SVM_L,20)
SVM_R_plot <- IMP(SVM_R,20)
## vis
ggbarplot(SVM_L_plot, x = 'top features', y = 'weight',
          fill = "lightblue", 
          xlab = "top features", ylab = "weight",
          sort.val = "desc", 
          top = 20,          
          x.text.angle = 45  
)

ggbarplot(SVM_R_plot, x = 'top features', y = 'weight',
          fill = "pink", 
          xlab = "top features", ylab = "weight",
          sort.val = "desc", 
          top = 20,          
          x.text.angle = 45  
)
