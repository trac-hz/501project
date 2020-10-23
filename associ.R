library(networkD3)
library(arules)
library(rtweet)
library(twitteR)
library(ROAuth)
library(jsonlite)
library(rjson)
library(tokenizers)
library(tidyverse)
library(plyr)
library(dplyr)
library(ggplot2)
library(syuzhet)
library(stringr)
library(arulesViz)
library(igraph)

#set the work directory, please change this to your own path
setwd("C:/Users/Yiyang/Desktop/501/assign")

#read in the ted talk tags file (csv) that are sampled from the original dataset
#original dataset contains 2551 rows
#rows with "math": 58
#rows with "education": 154
#rows with "teach": 44
#rows with any of the three above: 208

tags_DF <- read.csv("IMDB Dataset.csv", header=TRUE, sep=",")
TransactionTweetsFile = "TweetResults.csv"
Trans <- file(TransactionTweetsFile)

#tokenize to words
Tokens <- tokenizers::tokenize_words(tags_DF[1,1],
                                     stopwords = stopwords::stopwords('en'),
                                     lowercase = TRUE, strip_punct = TRUE,
                                     strip_numeric = TRUE, simplify = TRUE)

print(tags_DF[1,1])
# Write tokens
cat(unlist(Tokens), "\n", file=Trans, sep=",")
close(Trans)

# Append remaining lists of tokens into file
Trans <- file(TransactionTweetsFile, open = "a")
for(i in 2:1000){
        Tokens <- tokenize_words(tags_DF[i,1],stopwords = stopwords::stopwords("en"), 
                                 lowercase = TRUE,  strip_punct = TRUE,strip_numeric = TRUE,simplify = TRUE)
        cat(unlist(Tokens),"\n", file=Trans, sep=",")
}
close(Trans)

library(dplyr)
# Read in the tweet transactions
TweetDF <- read.csv(TransactionTweetsFile, 
                    header = FALSE, sep = ",")
head(TweetDF)
(str(TweetDF))

# Convert all columns to char 
TweetDF <- TweetDF %>%
        mutate_all(as.character)
(str(TweetDF))

# We can now remove certain words
TweetDF[TweetDF == "t.co"] <- ""
TweetDF[TweetDF == "rt"] <- ""
TweetDF[TweetDF == "http"] <- ""
TweetDF[TweetDF == "https"] <- ""

# Remove the duplicate tweets
TweetDF <- TweetDF[!duplicated(TweetDF),]

# Now we save the dataframe using the write table command 
write.table(TweetDF, file = "UpdatedTweetFile.csv", col.names = FALSE, 
            row.names = FALSE, sep = ",")

TweetTrans <- read.transactions("UpdatedTweetFile.csv", sep =",", 
                                format("basket"),rm.duplicates = TRUE)
inspect(TweetTrans)


# ARM
TweetTrans_rules = arules::apriori(TweetTrans, 
                                   parameter = list(support=.01, conf=.05,minlen=2))
inspect(TweetTrans_rules)

# Sort by Conf
SortedRules_conf <- sort(TweetTrans_rules, by="confidence", decreasing=TRUE)
inspect(SortedRules_conf)

# Sort by Sup
SortedRules_sup <- sort(TweetTrans_rules, by="support", decreasing=TRUE)
inspect(SortedRules_sup)

# Sort by Lift
SortedRules_lift <- sort(TweetTrans_rules, by="lift", decreasing=TRUE)
inspect(SortedRules_lift)

TweetTrans_rules<-SortedRules_sup[1:50]
inspect(TweetTrans_rules)

# We're also interested in the associations with "movie" on the left hand side!
movie_rules <- arules::apriori(TweetTrans,
                               parameter = list(support=.001, conf=.01, minlen=2),
                               appearance = list(default="rhs", lhs="movie"))
inspect(movie_rules)

# Sort by Conf
SortedMovieRules_conf <- sort(movie_rules, by="confidence", decreasing=TRUE)
inspect(SortedMovieRules_conf)

# Sort by Sup
SortedMovieRules_sup <- sort(movie_rules, by="support", decreasing=TRUE)
inspect(SortedMovieRules_sup)

# Sort by Lift
SortedMovieRules_lift <- sort(movie_rules, by="lift", decreasing=TRUE)
inspect(SortedMovieRules_lift)

# Visualize
library( RColorBrewer)
plot(SortedMovieRules_lift[1:15], method="graph")
plot(SortedRules_lift[1:15],method="graph")

plot(movie_rules, control = list(jitter = 1.5, col = rev(brewer.pal(9, "Blues")[4:9])), shading = "lift", cex = 0.3)
plot(TweetTrans_rules, control = list(jitter = 1.5, col = rev(brewer.pal(9, "Purples")[4:9])), shading = "lift", cex = 0.3)

####################  Using Network3D To View Results  ########################

# Convert the RULES to a DATAFRAME
Rules_DF <- DATAFRAME(TweetTrans_rules, separate = TRUE)
(head(Rules_DF))
str(Rules_DF)
## Convert to char
Rules_DF$LHS<-as.character(Rules_DF$LHS)
Rules_DF$RHS<-as.character(Rules_DF$RHS)

## Remove all {}
Rules_DF[] <- lapply(Rules_DF, gsub, pattern='[{]', replacement='')
Rules_DF[] <- lapply(Rules_DF, gsub, pattern='[}]', replacement='')

head(Rules_DF)

# Remove the sup, conf, and count
# USING LIFT
Rules_L<-Rules_DF[c(1,2,6)]
names(Rules_L) <- c("SourceName", "TargetName", "Weight")
head(Rules_L,30)

# USING SUP
Rules_S<-Rules_DF[c(1,2,3)]
names(Rules_S) <- c("SourceName", "TargetName", "Weight")
head(Rules_S,30)

# USING CONF
Rules_C<-Rules_DF[c(1,2,4)]
names(Rules_C) <- c("SourceName", "TargetName", "Weight")
head(Rules_C,30)

Rules_Sup<-Rules_L

# BUILD THE NODES & EDGES
(edgeList<-Rules_Sup)
MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE))

nodeList <- data.frame(ID = c(0:(igraph::vcount(MyGraph) - 1)), 
                       # because networkD3 library requires IDs to start at 0
                       nName = igraph::V(MyGraph)$name)
# Node Degree
(nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                       v = igraph::V(MyGraph), mode = "all")))

# Betweenness
BetweenNess <- igraph::betweenness(MyGraph, 
                                   v = igraph::V(MyGraph), 
                                   directed = TRUE) 

(nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))

# BUILD THE EDGES
# Recall that ... 
# edgeList<-Rules_Sup
getNodeID <- function(x){
        which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
}

edgeList <- plyr::ddply(
        Rules_Sup, .variables = c("SourceName", "TargetName" , "Weight"), 
        function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                                TargetID = getNodeID(x$TargetName)))

head(edgeList)
nrow(edgeList)

DiceSim <- igraph::similarity.dice(MyGraph, vids = igraph::V(MyGraph), mode = "all")
head(DiceSim)

# Create  data frame that contains the Dice similarity between any two vertices
F1 <- function(x) {data.frame(diceSim = DiceSim[x$SourceID +1, x$TargetID + 1])}
# Place a new column in edgeList with the Dice Sim
head(edgeList)
edgeList <- plyr::ddply(edgeList,
                        .variables=c("SourceName", "TargetName", "Weight", 
                                     "SourceID", "TargetID"), 
                        function(x) data.frame(F1(x)))
head(edgeList)

# color
COLOR_P <- colorRampPalette(c("#00FF00", "#FF0000"), 
                            bias = nrow(edgeList), space = "rgb", 
                            interpolate = "linear")
COLOR_P
(colCodes <- COLOR_P(length(unique(edgeList$diceSim))))
edges_col <- sapply(edgeList$diceSim, 
                    function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
nrow(edges_col)

# NetworkD3 Object
D3_network_Tweets <- networkD3::forceNetwork(
        Links = edgeList, # data frame that contains info about edges
        Nodes = nodeList, # data frame that contains info about nodes
        Source = "SourceID", # ID of source node 
        Target = "TargetID", # ID of target node
        Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship among nodes
        NodeID = "nName", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
        Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
        Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
        height = 700, # Size of the plot (vertical)
        width = 900,  # Size of the plot (horizontal)
        fontSize = 20, # Font size
        linkDistance = networkD3::JS("function(d) { return d.value*10; }"), # Function to determine distance between any two nodes, uses variables already defined in forceNetwork function (not variables from a data frame)
        linkWidth = networkD3::JS("function(d) { return d.value/10; }"),# Function to determine link/edge thickness, uses variables already defined in forceNetwork function (not variables from a data frame)
        opacity = 0.9, # opacity
        zoom = TRUE, # ability to zoom when click on the node
        opacityNoHover = 0.9, # opacity of labels when static
        linkColour = "red"   # "edges_col"red"# edge colors
) 

# Save network as html file
networkD3::saveNetwork(D3_network_Tweets, 
                       "NetD3(1).html", selfcontained = TRUE)

