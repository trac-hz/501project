## install script
if (!"devtools" %in% installed.packages()) {
  install.packages("devtools")
}
devtools::install_github("mkearney/newsAPI")

## load package
library(newsAPI)

## my obscured key
NEWSAPI_KEY <- "d31bf501afc242ed92c65a2736aa709c"

## save to .Renviron file
cat(
  paste0("NEWSAPI_KEY=", NEWSAPI_KEY),
  append = TRUE,
  fill = TRUE,
  file = file.path("~", ".Renviron")
)

src <- get_sources(language = "en")

## preview data
print(src, width = 500)

## apply get_articles function to each news source
df <- lapply(src$id, get_articles)

## collapse into single data frame
df <- do.call("rbind", df)

## preview data
print(df, width = 500)
