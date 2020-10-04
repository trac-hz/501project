# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 02:41:08 2020

@author: Yiyang
"""
import numpy as np  
import pandas as pd  
import os
os.chdir(r"C:\Users\Yiyang\Desktop\DUE\module2")

df=pd.read_csv('MovieCriticDave_tweets1.csv', encoding = "ISO-8859-1")
df.head()
df.describe().T

import unicodedata
import re

def normalize(text):
    text = unicodedata.normalize('NFKC',text.lower())
    text = re.sub(r"[\.\!\/\\_,$%^*()+\"\']+|[+鈥斺€�?:;<>~@#锟�&]+", " ", text)
    return text

import nltk

stemmer = nltk.stem.SnowballStemmer('english') 

def tokenize(text, remove_punct=True, stem_tokens=False):
    tokens = []
    for token in nltk.word_tokenize(text):
        if remove_punct and token in string.punctuation: 
            continue
        if stem_tokens:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return tokens

from nltk.corpus import stopwords
#nltk.download()
stopwords = stopwords.words('english')

import string
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer 

def get_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:          
        return None

lemmatizer = WordNetLemmatizer()

def removestopwords(text, remove_punct=False,  remove_stop=False, stem_tokens=False, lemmatize_tokens=False, all_fields=False):
    tokens = []
    text = normalize(text)
 
    for token in nltk.pos_tag(nltk.word_tokenize(text)):
        stem = ''
        token_text = token[0]
        token_pos = token[1]
        if remove_punct and token_text in string.punctuation: 
            continue
        if remove_stop and token_text.strip().lower() in stopwords:
            continue
        if stem_tokens or all_fields:
            stem = stemmer.stem(token_text)
        if lemmatize_tokens or all_fields:
            wordnet_tag = get_wordnet_tag(token_pos)
            if wordnet_tag is not None:
                lemma = lemmatizer.lemmatize(token_text,wordnet_tag)
            else:
                lemma = token_text
        if all_fields:
            tokens.append({'token': token_text, 'stem': stem, 'lemma': lemma})
        elif stem_tokens:
            tokens.append(stem)
        else:
            tokens.append(token_text)     
    return tokens
#nltk.download()
for i in range(0,len(df)):
    tweet = df.loc[i,'text']
    normalized_tweet = normalize(tweet)
    tokens = tokenize(normalized_tweet,remove_punct=False, stem_tokens=False)
    nostopwords = removestopwords(normalized_tweet,remove_punct=True, remove_stop=True, lemmatize_tokens=True)
    stem = removestopwords(normalized_tweet,remove_punct=True, remove_stop=True, stem_tokens=True, lemmatize_tokens=True)
    print("original: ",tweet)
    print("normalized: ",normalized_tweet)
    print("tokenized: ",tokens)
    print("removed stopwords: ",nostopwords)
    print("stemmed: ",stem)
    print()

from jieba import analyse
def get_frequency_words(file):
    with open(file, 'r') as f:
        texts = f.read()
        top_words = analyse.textrank(texts, topK=400, withWeight=True)
        ret_words = {}
        for word in top_words:
            ret_words[word[0]] = word[1]
    return ret_words
words_frequency = get_frequency_words(df)

import wordcloud
wordcloud.generate_from_text(text=df)
wordcloud.generate_from_frequencies(frequencies=words_frequency)
    
df.to_csv('cleanTwitter.csv',index=False) 