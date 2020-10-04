# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 02:33:25 2020

@author: Yiyang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
from wordcloud import WordCloud

df=pd.read_csv('movies.csv', encoding = "ISO-8859-1")
df.head()
df.describe().T
sns.heatmap(df.isnull(), cbar=False) 
plt.title('missing value by column', fontsize = 15)
plt.show()
# Check if there are missing observations
df.info()
#see quantative data statistics
for index in ['budget', 'gross', 'runtime', 'score', 'votes', 'year']:
    print(index, 'min =', df[index].min(), '&', index, 'max =', df[index].max())

