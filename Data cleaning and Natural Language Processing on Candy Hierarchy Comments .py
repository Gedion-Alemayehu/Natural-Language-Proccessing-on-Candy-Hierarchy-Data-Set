#!/usr/bin/env python
# coding: utf-8

# #### Halloween Candy Heirarchy

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import re
# Import a messy dataset (csv) with Pandas (5 pts)
# Import the data from Halloween Candy Heirarchy 2017
df = pd.read_csv('candyheirarchy2017_2.csv')
df.head()


# In[4]:


df.tail()


# ###### Clean the dataset with Pandas 

# ###### Exclude all but one country? Make country names uniform?
# 

# In[5]:


#change the name of columns for clarity and separate use
df.columns = [x.upper() for x in df.columns.str.replace(" ","_").str.replace("[|:]","")]
df.head()


# In[6]:


# to exclude all but one country first check how many countries are listed 
df.Q4_COUNTRY.unique()


# In[7]:


#drop all NAn values in Country 
df = df.dropna(subset = ['Q4_COUNTRY'])
# Exclude all the countries but the U.S.
df = df[(df.Q4_COUNTRY.str.contains('U',flags=re.IGNORECASE)) & (df.Q4_COUNTRY.str.contains('S', flags=re.IGNORECASE))]
# check if other countries are still in for more cleaning
df.Q4_COUNTRY.unique()


# In[8]:


# delete the other countries that got mixed in for final cleaning 
notUsa = ['Australia', 'australia','Trumpistan', 'South africa', 'South Korea', 'soviet canuckistan', 'subscribe to dm4uz3 on youtube']
df=df[~df['Q4_COUNTRY'].isin(notUsa)]
df.Q4_COUNTRY.unique()


# In[9]:


#make all country names uniform 
for index in df.index:
    if df.loc[index,'Q4_COUNTRY'] != 'USA':
        df.loc[index,'Q4_COUNTRY'] = 'USA'
df


# ###### Remove rows with high amounts of empty data?

# In[10]:


# remove rows based on column condition. # count how many columns there are
columnCount = df.shape[1]
print(columnCount)
#if 90 % of the column data is empty, NAn then drop the row
df1 = df.dropna(thresh = 0.9 * 120)
#check how many rows were dropped
originalRows = df.shape[0]
filteredRows = df1.shape[0]
print(originalRows)
print(filteredRows)
df1


# ###### Identify some aspect of the dataset to analyze and make a visualization of your analysis 

# In[11]:


## What was the favorite candy? 
# create a rating map
# give the the rating values a numerical substitute
rateMap = {"JOY":10, "MEH":5, "DESPAIR":0}
df1 = df1.applymap(lambda q: rateMap.get(q) if q in rateMap else q)
rateCandies = df1.iloc[ : , 6:-10 ]
#create the visuals
plt.figure(figsize=[20,10])
plt.xlabel("Rating")
plt.ylabel("Type of Candy")
plt.xlim(0,10)
plt.title("Most Popular Candies")
rateCandies.mean().sort_values(ascending=False).head(15).plot(kind='pie', autopct='%1.1f%%',fontsize=18, );


# 
# ###### Use the subjective/objective scale to identify the average sentiment in the other commnets section

# In[12]:


# delete all the nan values for precision 
df1 = df1.dropna(subset = ['Q9_OTHER_COMMENTS'])
# Source: https://data-science-blog.com/en/blog/2018/11/04/sentiment-analysis-using-python/
from textblob import TextBlob
df1['Q9_OTHER_COMMENTS'] = df1['Q9_OTHER_COMMENTS'].astype(str)
def sentipolarity(x):
    return TextBlob(x).sentiment.polarity 
df1['senti_polarity'] = df1['Q9_OTHER_COMMENTS'].apply(sentipolarity)
df1.senti_polarity = df1.senti_polarity.replace(0, np.NaN)
print(df1.senti_polarity.mean())
df1.hist(column='senti_polarity')


# Discuss Results:
#     The mean polarity and the graph show that the comments were more or less positive. 

# ###### Make a wordcloud of Q9: OTHER COMMENTS (5
# 

# In[13]:


from wordcloud import WordCloud, STOPWORDS 
#concacenate all the words in the column
wordcloud = WordCloud().generate(' '.join(df1['Q9_OTHER_COMMENTS']))
#do the visualization for the wordcloud 
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# Discuss your results:
# We can infer from all the visualization that we did that the comments provided in Q9 were more or less positive. The wordcloud gives us a good visual to prove just that, it portrays the most common words used in the concacenated list of comments and the bigger words are the most highlighted: candy, chocolate, one, trick, halloween, chocolate...
