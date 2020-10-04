library(ggplot2)
df<-read.csv('C:/Users/Yiyang/Desktop/DUE/module2/movies.csv')
#quantative data
#Budget, Gross, Runtime, Score and Votes.

ggplot(df) +
  aes(x = budget) +
  geom_histogram(bins = 30L, fill = "#0c4c8a") +
  theme_minimal()
##boxplot for gross
ggplot(df) +
  aes(y = gross ) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

#extract row number
out1 <- boxplot.stats(df$gross)$out
out_ind1 <- which(df$gross %in% c(out1))
out_ind1

#remove lines
library(dplyr)
df2 <- df %>% slice(-out_ind1)
ggplot(df2) +
  aes(x = "", y = gross ) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()



