# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:58:09 2020

@author: Yiyang
"""
import requests
import json
BaseURL="http://newsapi.org/v2/everything"

URLPost={"apiKey":"d31bf501afc242ed92c65a2736aa709c",
         "q":"movie",
         "from":"2020-09-02",
         "to":"2020-10-02",
         "sortBy":"popularity"}

response=requests.get(BaseURL,URLPost)
jsontxt=response.json()
file=open("NewsAPI_output.txt","a")
print(jsontxt,file=file)
file.close()


art=jsontxt["articles"]
