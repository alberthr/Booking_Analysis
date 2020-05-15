#!/usr/bin/env python
# coding: utf-8

# ### GOAL
# The main purpose of this notebook is:  
# 1- Make an analysis of our variables from different perspectives.  
# 2- Confirm that our dataset has fair information based on our business knowledge.  
# 3- Understand correlation between variables.  
# 4- if it is necessary create new variables or modify some information like nulls.  
# 5- Understand better the information to implement the right models.

# In[1]:


import datetime as dt
import math # funciones matematicas
from typing import Tuple, List, Dict, Set
from datetime import datetime # fechas
import numpy as np # funciones matemáticas para operar con vectores o matrices
import random # generador de datos aleatorios
from matplotlib import pyplot as plt # gráficos
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D # gráficos 3D
from matplotlib import cm # mapa de colores
import seaborn as sns # gráficos más avanzados
import pandas as pd # analisis y manipulación y filtrado de datos, creación de data.frames(tablas)
import scipy.stats # Machine learning
import requests # Apis
import json
import scipy.stats # estadística y distribuciones de probabilidad
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose ## Descomposición
from statsmodels.tsa.arima_model import ARIMA
import geopandas


# In[2]:


df = pd.read_csv('/root/TFM/Data/df_features')


# In[3]:


df['Review_Date'] = df['Review_Date'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))


# In[4]:


df['Price'].fillna(0)


# In[5]:


df['Reservation_ADR'] = df['Price'] * df['Length_N']


# In[6]:


df['Reservation_ADR'].fillna(0)


# ### Variable definition: 
# Includes variables from the dataset and new ones created in order to feed our dataset with more relevant information.

# In[7]:


df.columns


# 1-Hotel_Address: Address of hotel. (type: String)  
# 2-Additional_Number_of_Scoring: There are also some guests who just made a scoring on the service rather than a review. This number indicates how many valid scores without review in there. (type: integer)  
# 3-Review_Date: Date when reviewer posted the corresponding review. (type: Date)  
# 4-Average_Score: Average Score of the hotel, calculated based on the latest comment in the last year.(type: float)  
# 5-Hotel_Name: Name of Hotel. (type: String)  
# 6-Reviewer_Nationality: Nationality of Reviewer. (type: String)  
# 7-Negative_Review: Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Negative'. (type: string)  
# 8-Total_Number_of_Reviews: Total number of valid reviews the hotel has. (type: integer)  
# 9-Review_Total_Negative_Word_Counts: Total number of words in the negative review.(type: integer)  
# 10-Positive_Review: Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Positive'. (type: string)  
# 11-Review_Total_Positive_Word_Counts: Total number of words in the positive review.(type: integer)  
# 12-Total_Number_of_Reviews_Reviewer_Has_Given: Number of Reviews the reviewers has given in the past. (type: integer)  
# 13-Reviewer_Score: Score the reviewer has given to the hotel, based on his/her experience.(type: float)  
# 14-days_since_review: Duration between the review date and scrape date.(type: integer)  
# 15-lat: Latitude of the hotel.(type: float)  
# 16-lng: longtitude of the hotel. (type: float)  
# 17- Diff: This variable was not included in the original dataset. Difference betwee Avrage Score and Customer review score (type: float)   
# 18- Review_Month: month of the review score transformed into a number (type: integer)  
# 19- Review_Year: year of the review score transformed into a number (type: integer)  
# 20- Country: Country where hotel is located (type: string)  
# 21- City: City where hotel is located (type: string)  
# 22- Pet: Travell with pet  (type: string)  
# 23- Purpose: reason for travelling (type: string)  
# 24- Whom: Customer profile. (type: string)  
# 25- Room: Room type. (type: String)  
# 26- Length: it represents the number of nights stayed by the customer. (type: String)  
# 27- Device: Reservation made by mobile or desktop. (type: String)  
# 28- Room_Recode: Room type clusterized in groups. This ratio was not included in the original dataset. (type: String)  
# 29- Nationality_Recode: Customer nationality clusterized in groups based on those world regions that use to provide similar review scores. (type: String)  
# 30- Length_Recode: It represents the number of nights stayed by the customer clusterized in gropus from 1 to 9 nights.(type: String)  
# 31- Close_Landmarks: This variable was not included in the original dataset. It represent how many of the top 10 landmarks of the city are close to the hotel. Location is key and cusomers, travelling for leisure, not only find important to be close to a specific landmark but also to a multiple landamarks. We take as a reference the center point and closest point to these top 10 landamarks. (type: float)  
# 32- Dist_Center: This variable was not included in the original dataset. Distance from hotel to city center (type: float)  
# 33- Dist_Airport: This variable was not included in the original dataset. Distance from hotel to main Airport (type: float)  
# 34- Dist_Train: This variable was not included in the original dataset. Distance from hotel to main train station (type: float)
# 35- Price: This variable was not included in the original dataset. We do not know  the real price of the reservation. Based on scrapping process we extract a reference price provided by Booking.com. (type: float)  
# 36- Stars: number of hotel stars. (type: Strings)  
# 37- Lenght_N: it represents the number of nights stayed by the customer. (type: String)  
# 38- Reservation_ADR: This variable was not included in the original dataset. It represents the total cost of the reservation. We created this ratio because it is a clear influencer during the evaluation process in terms of value for money. (type: float) 

# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


df.count()


# ### UNIVARIABLE ANALYSIS: Quantitative metrics
# We want to understand the distribution and the impact of the outlayers. For instance, comparing difference between means and medians.

# In[11]:


# Additional Number of Scoring: Some hotels have a big number of reviews apart from those included in the Reviewer score 
# column. It moves our mean to the right in relation to our median.

fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Additional_Number_of_Scoring'], bins = 25, label = 'Addittional number of scoring')
ax[0].axvline(df['Additional_Number_of_Scoring'].mean(), color = 'r')
ax[1].boxplot(df['Additional_Number_of_Scoring'],vert = False)
ax[0].legend()


# #### Most Hotel Average Scores are between 8 and 9.

# In[94]:


fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Average_Score'], bins = 25, label = 'Hotel Average Score')
ax[0].axvline(df['Average_Score'].mean(), color = 'r')
ax[1].boxplot(df['Average_Score'],vert = False)
ax[0].legend()


# In[13]:


#Review Total Negative Word Counts: some negative reviews move the mean to the right

fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Review_Total_Negative_Word_Counts'], bins = 25, label = 'Negative words per comment')
ax[0].axvline(df['Review_Total_Negative_Word_Counts'].mean(), color = 'r')
ax[1].boxplot(df['Review_Total_Negative_Word_Counts'],vert = False)
ax[0].legend()


# In[14]:


# Total Number of Reviews per Hotel. Some hotels have a big number of reviews. It moves our mean to the right in relation to 
# our median.

fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Total_Number_of_Reviews'], bins = 25, label = 'Number of Reviews per Hotel')
ax[0].axvline(df['Total_Number_of_Reviews'].mean(), color = 'r')
ax[1].boxplot(df['Total_Number_of_Reviews'],vert = False)
ax[0].legend()


# In[15]:


# Review Total Positive Word Counts

fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Review_Total_Positive_Word_Counts'], bins = 100, label = 'Positive words per comment')
ax[0].axvline(df['Review_Total_Positive_Word_Counts'].mean(), color = 'r')
ax[1].boxplot(df['Review_Total_Positive_Word_Counts'],vert = False)
ax[0].legend()


# In[16]:


# Total Number of Reviews Reviewer Has Given

fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Total_Number_of_Reviews_Reviewer_Has_Given'], bins = 50, label = 'Number_of Reviews Reviewer has given')
ax[0].axvline(df['Total_Number_of_Reviews_Reviewer_Has_Given'].mean(), color = 'r')
ax[1].boxplot(df['Total_Number_of_Reviews_Reviewer_Has_Given'],vert = False)
ax[0].legend()


# In[17]:


# Reviewer Score

fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Reviewer_Score'], bins = 20, label = 'Reviewer Score')
ax[0].axvline(df['Reviewer_Score'].mean(), color = 'r')
ax[1].boxplot(df['Reviewer_Score'],vert = False)
ax[0].legend()


# In[18]:


# Diff: Most values are positive and will help hotel to increase its average score. However, there is
# a long cue in the negative numbers because some reviews are very negative moving the mean to the left.

fig, ax=plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(df['Diff'], bins = 25, label = 'Diff')
ax[0].axvline(df['Diff'].mean(), color = 'r')
ax[1].boxplot(df['Diff'],vert = False)
ax[0].legend()


# In[19]:


# Distance to Close_Landmarks

fig, ax=plt.subplots(1, 1, figsize=(10, 5))
ax.hist(df['Close_Landmarks'], bins = 10, label = 'Close_Landmarks')
ax.axvline(df['Close_Landmarks'].mean(), color = 'r')
ax.legend()


# In[20]:


# Distance to City center

fig, ax=plt.subplots(1, 1, figsize=(10, 5))
ax.hist(df['Dist_Center'], bins = 30, label = 'Dist_Center')
ax.axvline(df['Dist_Center'].mean(), color = 'r')
ax.legend()


# In[21]:


# Distance to Airport

fig, ax=plt.subplots(1, 1, figsize=(10, 5))
ax.hist(df['Dist_Airport'], bins = 30, label = 'Dist_Airport')
ax.axvline(df['Dist_Airport'].mean(), color = 'r')
ax.legend()


# In[22]:


# Price

fig, ax=plt.subplots(1, 1, figsize=(10, 5))
ax.hist(df['Price'], bins = 50, label = 'Price')
ax.axvline(df['Price'].mean(), color = 'r')
ax.legend()


# In[23]:


# Length_N

fig, ax=plt.subplots(1, 1, figsize=(10, 5))
ax.hist(df['Length_N'], bins = 20, label = 'Length_N')
ax.axvline(df['Length_N'].mean(), color = 'r')
ax.legend()


# In[24]:


# Reservation ADR 

fig, ax=plt.subplots(1, 1, figsize=(10, 5))
ax.hist(df['Reservation_ADR'], bins = 50, label = 'Reservation_ADR')
ax.axvline(df['Reservation_ADR'].mean(), color = 'r')
ax.legend()


# ### UNIVARIABLE ANALYSIS: Qualitative metrics
# 
# We want to undersand what qualitative characteristics are the most frequent in our reservation. We conclude that hotels located in London generate the same number of customer reviews than the rest of the 5 cities together. Most of the reservations were made by couples & solo travellers, from UK & Ireland, that use to be between 1 and 3 nights in a standard or deluxe room.

# In[25]:


num = df.Country.value_counts()
count_country= pd.DataFrame(num)

num1 = df.City.value_counts()
count_city= pd.DataFrame(num1)

pets_count = df.Pet.value_counts()
purpouse_count = df.Purpose.value_counts()
whom_counts = df.Whom.value_counts()
Room_type_counts = df.Room_Recode.value_counts().head()
LOS = df.Length.value_counts().head(7)
device_counts = df.Device.value_counts()
Reviewer_Nationality_counts = df.Nationality_Recode.value_counts().head()


# In[26]:


fig, ax = plt.subplots(9, 1, figsize = (12, 20))
ax[0].bar(count_country.index, count_country['Country'], label = 'Country')
ax[1].bar(count_city.index, count_city['City'], color='orange', label = 'City')
ax[2].bar(df['Pet'].value_counts().index.values, pets_count, color = 'green', label = 'Pet')
ax[3].bar(df['Purpose'].value_counts().index.values, purpouse_count, color = 'purple', label = 'Purpouse')
ax[4].bar(df['Whom'].value_counts().index.values, whom_counts, color = 'red', label = 'Whom')
ax[5].bar(Room_type_counts.index.values, Room_type_counts, color = 'magenta', label = 'Room Type')
ax[6].bar(LOS.index.values, LOS, color = 'brown', label = 'LOS')
ax[7].bar(device_counts.index.values, device_counts, color = 'darkblue', label = 'Device')
ax[8].bar(Reviewer_Nationality_counts.index.values, Reviewer_Nationality_counts, color = 'lightblue', label = 'Reviewer Nationality')
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
ax[5].legend()
ax[6].legend()
ax[7].legend()
ax[8].legend()


# ### MULTIVARIABLE ANALISYS - Quantitative & Qualitative metrics
# 
# #### Quantitative:  
# 'Additional_Number_of_Scoring', 'Review_Date', 'Average_Score',  'Reviewer_Nationality', 'Review_Total_Negative_Word_Counts','Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score', 'Diff', 'Review_Month', 'Review_Year',  
# 'days_since_review', 'lat', 'lng', 'Close_Landmarks', 'Dist_Center', 'Dist_Airport','Dist_Train',  
# 'Price', 'Length_N', 'Reservation_ ADR'  
# 
# #### Qualitative:  
# 'Hotel_Address', 'Hotel_Name', 'Review_Date', 'Negative_Review', 'Positive_Review',  
# 'Country', 'City','Pet', 'Purpose', 'Whom', 'Room_Recode',  'Length', 'Length_Recode',  
# 'Nationality_Recode', 'Device', , 'Review_Year', 'Stars'
# 

# ### Which are the variables with the highest correlation?  
# #### 1- Hotels with more reviews uses to get a higher additional number of reviews.    

# In[27]:


plt.scatter(df['Additional_Number_of_Scoring'], df['Total_Number_of_Reviews'])


# In[89]:


print('Cor Reviewer Score & Difference Score is', np.corrcoef(df['Reviewer_Score'], df['Diff']))


# #### 2- The highest is the reviewer Score, highest is the difference between Reviewer_Score and Average_Score.

# In[28]:


plt.scatter(df['Reviewer_Score'], df['Diff'])


# In[90]:


print('Cor Additional number of scoring & Total number of reviews is', np.corrcoef(df['Additional_Number_of_Scoring'], df['Total_Number_of_Reviews']))


# #### 3- The closest is the hotel to the city center the closest is the hotel to the train station.

# In[91]:


plt.scatter(df['Dist_Center'], df['Dist_Train'])


# #### 4- Those hotels closer to the city center generates booking with higher Reservation_ADR

# In[87]:


plt.scatter(df['Dist_Center'], df['Reservation_ADR'])


# #### 5- Reservations with more number of nights use to be more expensive.  

# In[92]:


plt.scatter(df['Length_N'], df['Reservation_ADR'])


# #### Let's take a look to the rest of quantitative variables with lower correlation. 

# In[32]:


sns.pairplot(df.drop(columns=['Hotel_Address', 'Review_Date', 'Hotel_Name', 'Reviewer_Nationality', 'Negative_Review',
                              'Positive_Review', 'lat', 'lng', 'Length', 'Review_Month', 'Review_Year', 'Country', 'City',
                             'Pet', 'Purpose', 'Whom', 'Room','Room_Recode', 'Length_Recode', 'Nationality_Recode', 'Device', 'Close_Landmarks', 'Stars']))


# ### Is there any relation between hotel Average_Score and negative or positive number of words of customer reviews?
# 
# Those hotels with lower average score have reviews with more negative words.  
# Those hotels with higher average score have reviews with more positive words.  
# Check below grid and graph.

# In[33]:


fig, ax = plt.subplots(1,1, figsize=(14,5))
ax.scatter(df['Average_Score'], df['Review_Total_Negative_Word_Counts'], c = 'r', alpha = .2, label = 'Negative')
ax.scatter(df['Average_Score'], df['Review_Total_Positive_Word_Counts'], c = 'b', alpha = .2, label = 'Positive')
ax.legend()


# In[34]:


df[['Average_Score', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']].groupby('Average_Score').mean().sort_values(
        ('Review_Total_Negative_Word_Counts'), ascending=True)


# #### We find the same conclusion between the Reviewer_Score and the number of positive and negative words.

# In[35]:


fig, ax = plt.subplots(1,1, figsize=(14,5))
ax.scatter(df['Reviewer_Score'], df['Review_Total_Negative_Word_Counts'], c = 'r', alpha = .2, label = 'Negative')
ax.scatter(df['Reviewer_Score'], df['Review_Total_Positive_Word_Counts'], c = 'b', alpha = .2, label = 'Positive')
ax.legend()


# In[36]:


df[['Reviewer_Score', 'Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts']].groupby('Reviewer_Score').mean().sort_values(
        ('Review_Total_Negative_Word_Counts'), ascending=False)


# ### Nationalities
# We want to understand if different nationalities tend to get similar scores impacting in the hotel Average_Score postive or negatively. Based on results we are going to create regions to simplify our data.  
# Let's take a look at those regions with similar Diff (difference between Reviewer_Score and Average_Score).  
# Map below shows those regions that use to score over (red) and below (blue) hotel average.

# In[37]:


diff_country = df[['Reviewer_Nationality','Diff']].groupby('Reviewer_Nationality').mean().reset_index()
diff_country.columns = ['name','Diff']
diff_country.name = diff_country.name.apply(lambda x: x[1:-1])


# In[38]:


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))


# In[39]:


tmp = world.merge(diff_country, on='name', how='inner')


# In[40]:


fig, ax = plt.subplots(figsize=(15,6))
#tmp.plot(column='Diff', ax=ax, legend = True, cmap='coolwarm', scheme='box_plot')
tmp.plot(column='Diff', ax=ax, legend = True, cmap='coolwarm', scheme='quantiles')
plt.tight_layout()


# #### Let's see which regions generates higher and lower Diff / Reviewer_Score.  
# 1-Bookings from North America, Oceania and UK & Ireland are those with higher Diff and Reviewer_Score, helping hotels to improve their Average_Score.  
# 2-However, those bookings from  Middle East, Arabian Countries and Asia & Pacific have a more negative impact in the Average Score despite of getting an average Reviewer_Score between 7.9 and 8.1.   

# In[41]:


avg_regions = df[['Diff', 'Nationality_Recode', 'Reviewer_Score'
                  ]].groupby('Nationality_Recode').describe().sort_values(
                      ('Diff', 'mean'), ascending=False)

avg_regions


# #### Middle East and Arabian Countries pay more money for an accommodation and European travelers pay less money than the rest of the world. 

# In[42]:


df[['Reservation_ADR', 'Nationality_Recode'
    ]].groupby('Nationality_Recode').describe().sort_values(
        ('Reservation_ADR', 'mean'), ascending=False).head(10)


# In[43]:


p = pd.pivot_table(df[['Reservation_ADR', 'Nationality_Recode', 'Room_Recode']],
               values=('Reservation_ADR'), index=['Nationality_Recode'], columns=['Room_Recode'],
               aggfunc=np.mean)
p


# #### Relation between Nationality and customer opinion
# The number of negative words used in negative reviews is similar in all regions, between 15 and 19 words. However, positive reviews have a wider range of number of words, between 13 and 22.  
# It seems to be a tendency to be slighlty more descriptive when the review is negative.This aspect is going to be analyze more in depth in other notebooks.

# In[44]:


df['Review_Total_Positive_Word_Counts'].mean()


# In[45]:


df[['Review_Total_Positive_Word_Counts', 'Nationality_Recode'
    ]].groupby('Nationality_Recode').describe().sort_values(
        ('Review_Total_Positive_Word_Counts', 'mean'), ascending=False).head(10)


# In[46]:


df['Review_Total_Negative_Word_Counts'].mean()


# In[47]:


df[['Review_Total_Negative_Word_Counts', 'Nationality_Recode'
    ]].groupby('Nationality_Recode').describe().sort_values(
        ('Review_Total_Negative_Word_Counts', 'mean'), ascending=False).head(10)


# ### Which cities get better scores? 
# Hotels with more reviews provide a more objective score. More reviews and higher score means more credibility and becomes more atractive for customers. We understand that those hotels with at least 100 reviews have enough number of reviews.  
# 
# We want to know how many hotels have at least 100 reviews and the result is 1074 out of 1500. It means that most hotels of the 6 cities analyzed have a reliable score. 

# In[48]:


sum(df.Hotel_Address.value_counts() > 100)


# In[49]:


count_reviews_hotel = pd.DataFrame(
    df.groupby(['Hotel_Name', 'City', 'Average_Score']).apply(lambda x: len(x)).reset_index())

count_reviews_hotel.rename(columns={0: "Reviews"}, inplace=True)

count_reviews_hotel[count_reviews_hotel.Reviews > 100].sort_values('Average_Score', ascending=False).head(10)


# #### Barcelona and Viena are the cities where hotels have a higher Average Score 

# In[50]:


df[['City', 'Average_Score']].groupby('City').describe()


# In[51]:


fig, ax = plt.subplots(figsize=(15,5))
sns.violinplot(ax=ax, x="City", y="Average_Score", data=df, palette="Pastel1")


# #### Let's repeat the exercise but only taking into account each hotel once to align hotels with high and low number of reviews

# In[52]:


hotel_average = df[['Hotel_Name','City','Average_Score']].groupby(['Hotel_Name','City']).mean().reset_index()


# In[53]:


hotel_average[['City', 'Average_Score']].groupby('City').describe()


# In[54]:


fig, ax = plt.subplots(figsize=(15,5))
sns.violinplot(ax=ax, x="City", y="Average_Score", data=hotel_average, palette="Pastel1")


# #### Paris is the most expensive city. However, it is not the city with  the lowest Reviewer_Score. There are more aspects than price that influence customer's opinion

# In[55]:


df[['Reservation_ADR', 'City', 'Reviewer_Score'
    ]].groupby('City').describe().sort_values(
        ('Reservation_ADR', 'mean'), ascending=False)


# In[56]:


df[['Price', 'City'
    ]].groupby('City').describe().sort_values(
        ('Price', 'count'), ascending=False)


# ### Seasonality: time series with Reviewer_Score, Length_N & Reservation_ADR
# 
# We find a clear seasonality in the dataset when we analyze the Reviewer_Score, Lenght_N and Reservation_ADR.  
# Summer is the period of the year when people stay more days (Length_N) and spend more money (Reservatian_ADR).  
# Customers are more demanding giving lower reviews compared to the low season when it get again a higher value.  
# We can find some pick events during the year such as Easter and new year's Eve.

# #### 1- Reviewer_Score

# In[57]:


#1 There's a seasonality with lower scores around october and higher values at the beginning of the year
df[['Review_Date','Reviewer_Score']].groupby('Review_Date').mean().plot(figsize=(15,5))


# In[58]:


Reviews_Month = df[['Review_Month', 'Review_Year',
                    'Reviewer_Score']].groupby(['Review_Year',
                                                'Review_Month']).count()


# In[59]:


Reviews_Month.reset_index(inplace=True)
Reviews_Month


# #### 2- Length of stay

# In[60]:


df[['Review_Date','Length_N']].groupby('Review_Date').mean().plot(figsize=(15,5))


# In[61]:


pd.crosstab(df['Purpose'], df['Length_N'], normalize = True, margins= True)


# #### 3- Reservation_ADR

# In[62]:


df[['Review_Date','Reservation_ADR']].groupby('Review_Date').mean().plot(figsize=(15,5))


# ### Customer trends 

# #### Trip purpose  
# 
# Business Bookers are more demanding than those travelling for Leisure.  
# Leisure and Business bookers prefer Barcelona and Vienna.

# In[63]:


df[['Reviewer_Score', 'Purpose']].groupby('Purpose').mean()


# In[64]:


pd.pivot_table(df[['Reviewer_Score', 'Purpose', 'City']],
               values='Reviewer_Score', index=['Purpose'], columns=['City'],
               aggfunc=np.mean)


# In[65]:


Fig, ax= plt.subplots(1, 1, figsize = (12,8))
plt.bar(['Amsterdam', 'Barcelona', 'London', 'Milan', 'Paris', 'Vienna'], [8.526717, 8.607587, 8.435162, 8.446300, 8.496723, 8.605697], label='Leisure Trip', alpha = 0.3)
plt.bar(['Amsterdam', 'Barcelona', 'London', 'Milan', 'Paris', 'Vienna'], [8.044043, 8.154367, 7.867049, 7.985181, 8.077674, 8.292681], label='Business Trip',alpha = 0.8)
plt.legend()


# In[66]:


n = pd.pivot_table(df[['Diff', 'Purpose', 'City']],
               values='Diff', index=['Purpose'], columns=['City'],
               aggfunc=np.mean)
n


# #### How many days do customer use to stay?
# 82,09% of the reservations stay between 1-3 nights. 

# In[67]:


d = np.cumsum(df.Length_Recode.value_counts()) / len(df.Length)
d


# In[68]:


fig, ax=plt.subplots(1, 1, figsize=(10, 5))
ax.hist(d, bins=10, normed=True, cumulative=True)
fig.show()


# #### Couple & Solo traveller
# Couple and Solo travellers are the most frequet travellers and most of their bookings stay from 1 to 3 nights. However, Solo traveller is the customer that concentrates more reservations for just 1 night: 45,06%. Rest of customer have a better balance between 1-3 nights.

# In[69]:


pd.crosstab(df['Whom'], df['Length_Recode'], normalize ='index', margins= True)


# #### Solo_Traveller, is the most sensitive customer to hotel price? Does it influence on the review score?
# Solo_Traveller is the customer that gives lower scores despite of paying less money than the rest of customers. 
# However, room cost (Reservation_ADR) for a Solo_Traveller is similar to a couple despite of being just one person in the room. 

# In[70]:


df[['Reviewer_Score', 'Reservation_ADR', 'Whom']].groupby('Whom').mean()


# In[71]:


p = pd.pivot_table(df[['Reservation_ADR', 'Whom', 'Room_Recode', 'Reviewer_Score']],
               values=('Reservation_ADR', 'Reviewer_Score'), index=['Room_Recode'], columns=['Whom'],
               aggfunc=np.mean)
p


# #### Based on Reviewer_Score, Barcelona and Viena are their favourite cities and the cheapest for a Solo traveller. It does not happen in other customer profiles 

# In[72]:


t= pd.pivot_table(df[['Reviewer_Score','Whom', 'City']],
               values='Reviewer_Score', index=['Whom'], columns=['City'],
               aggfunc=np.mean)
t


# In[73]:


t.plot(figsize=(13,5))


# In[74]:


pd.pivot_table(df[['Reservation_ADR','Whom', 'City']],
               values='Reservation_ADR', index=['Whom'], columns=['City'],
               aggfunc=np.mean)


# In[75]:


s = pd.pivot_table(df[['Reservation_ADR', 'Whom', 'City']],
               values='Reservation_ADR', index=['Whom'], columns=['City'],
               aggfunc=np.mean)


# In[76]:


s.plot(figsize=(13,5))


# #### Purpose
# Main reason for travelling is leisure independently of the customer profile.  
# Solo travellers is the only one that combines frequently both, business and Leisure.

# In[77]:


pd.crosstab(df['Whom'], df['Purpose'], normalize= 'index', margins= True)


# In[78]:


figsize, ax=plt.subplots(1,1, figsize=(10, 10))
plt.bar(['Business trip', 'Leisure trip'],[0.2840, 0.463777], label='Couple', color ='darkblue')
plt.bar(['Business trip', 'Leisure trip'],[0.001432, 0.049323], label='Family with older children', color='red')
plt.bar(['Business trip', 'Leisure trip'],[0.004514, 0.112858], label='Family with young children', color = 'yellow')    
plt.bar(['Business trip', 'Leisure trip'],[0.014497, 0.112880], label='Group', color = 'orange')
plt.bar(['Business trip', 'Leisure trip'],[0.116391, 0.091754], label='Solo traveler', color = 'purple')
plt.bar(['Business trip', 'Leisure trip'],[0.000401, 0.003767], label='Travelers with friends', color= 'green')
plt.legend(loc='upper left')


# #### Travelling with Pets
# 
# Hotels located in Barcelona are the most valued for customers travelling with pets.
# 

# In[79]:


pd.pivot_table(df[['Reviewer_Score', 'Pet', 'City']],
               values='Reviewer_Score', index=['Pet'], columns=['City'],
               aggfunc=np.mean)


# In[80]:


Fig, ax= plt.subplots(1, 1, figsize = (8,8))
plt.bar(['Amsterdam', 'Barcelona', 'London', 'Milan', 'Paris', 'Vienna'], [8.42377, 8.598286, 8.287699, 8.332796, 8.435075, 8.447907], label='Reviewer Score with a pet')
plt.legend()


# In[81]:


p = pd.pivot_table(df[['Reservation_ADR', 'Length_Recode', 'Room_Recode', 'Reviewer_Score']],
               values=('Reservation_ADR', 'Reviewer_Score'), index=['Room_Recode'], columns=['Length_Recode'],
               aggfunc=np.mean)
p


# #### Next steps:
# 
# We have analyzed the different qualitative and quantitative variables of the dataset. We mixed variables from the original dataset with new ones created based on our business knowledge.  
# We tried to understand the impact of the outlayers, use our business knowledge to confirm customer trends (seasonality, customer profiles, booking characterics) that can have an influence on customer mindset and its customer review.  
# Now, we we need to combine all this variables to predict a key Metric: "Diff" (difference between customer review and hotel average Score). It will help hotels to react on time paying special attention on those potential negative reviews.
