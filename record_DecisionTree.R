library(rpart)
library(rattle)
###PREPARATION-----------------------------------------------

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

## ---------------------Now we can use decision tree--------------------:
## Decision trees:
Test_label <- Test$score_cat
Train_label <- Train$score_cat
Test_no_cat<-Test[,-c(6)]
Treefit <- rpart(score_cat ~ . , 
                 data = Train, method="class")
summary(Treefit)
fancyRpartPlot(Treefit, tweak=1.5)
predicted= predict(Treefit, Test_no_cat, type="class")
Results <- data.frame(Predicted=predicted,Actual=Test_label)
table(Results)
table_mat <-table(Results)
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test of all variables', accuracy_Test))


## Without runtime
Train1<-Train[,-c(5)]
Test1<-Test[,-c(5)]
Test_no_cat<-Test1[,-c(6)]
Test_label <- Test1$score_cat

Treefit1 <- rpart(score_cat ~ . , 
                  data = Train1, method="class")
summary(Treefit1)
fancyRpartPlot(Treefit1, tweak=1.3)

predicted1= predict(Treefit1, Test_no_cat, type="class")
Results1 <- data.frame(Predicted=predicted1,Actual=Test_label)
table(Results1)
table_mat1 <-table(Results1)
accuracy_Test1 <- sum(diag(table_mat1)) / sum(table_mat1)
print(paste('Accuracy for test of all variables without runtime', accuracy_Test1))

## Without genre
Train2<-Train[,-c(2)]
Test2<-Test[,-c(2)]
Test_no_cat<-Test2[,-c(6)]
Test_label <- Test2$score_cat

Treefit2 <- rpart(score_cat ~ . , 
                  data = Train2, method="class")
summary(Treefit2)
fancyRpartPlot(Treefit2, tweak=1.3)

predicted2= predict(Treefit2, Test_no_cat, type="class")
Results2 <- data.frame(Predicted=predicted2,Actual=Test_label)
table(Results2)
table_mat2 <-table(Results2)
accuracy_Test2 <- sum(diag(table_mat2)) / sum(table_mat2)
print(paste('Accuracy for test of all variables without genre', accuracy_Test2))

###########################################################################
## Without gross
Train3<-Train[,-c(3)]
Test3<-Test[,-c(3)]
Test_no_cat<-Test3[,-c(6)]
Test_label <- Test3$score_cat
Treefit3 <- rpart(score_cat ~ . , 
                  data = Train3, method="class")

summary(Treefit3)
fancyRpartPlot(Treefit3, tweak=1.3)

predicted3= predict(Treefit3, Test_no_cat, type="class")
Results3 <- data.frame(Predicted=predicted3,Actual=Test_label)
table(Results3)

## Without gross
Train4<-Train[,-c(4)]
Test4<-Test[,-c(4)]
Test_no_cat<-Test4[,-c(6)]
Test_label <- Test4$score_cat
Treefit4 <- rpart(score_cat ~ . , 
                  data = Train4, method="class")

summary(Treefit4)
fancyRpartPlot(Treefit4, tweak=1.3)

predicted4= predict(Treefit4, Test_no_cat, type="class")
Results4 <- data.frame(Predicted=predicted4,Actual=Test_label)
table(Results4)

