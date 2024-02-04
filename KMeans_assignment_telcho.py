# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 00:00:56 2023

@author: Vaibhav Bhorkade

"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:26:41 2023

@author: Vaibhav Bhorkade

Assignment - Telco_customer_churn.xlsx 
"""

"""
⦁ Perform clustering analysis on the telecom data set. 
The data is a mixture of both categorical and numerical data. 
It consists of the number of customers who churn out. Derive insights 
and get possible information on factors that may affect the churn decision.
 Refer to Telco_customer_churn.xlsx dataset
"""
"""
Business Objective 
Minimize : charges
Maximaze : Offers

Business constraints 
"""
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("C:/datasets/Telco_customer_churn.xlsx")
print(df)

# Data Dictionary
"""
                 Name_of_feature        Relevance
0                         Customer ID  Irrevance
1                               Count  Relevance
2                             Quarter  Relevance
3                   Referred a Friend  Relevance
4                 Number of Referrals  Relevance
5                    Tenure in Months  Relevance
6                               Offer  Relevance
7                       Phone Service  Relevance
8   Avg Monthly Long Distance Charges  Relevance
9                      Multiple Lines  Relevance
10                   Internet Service  Relevance
11                      Internet Type  Relevance
12            Avg Monthly GB Download  Relevance
13                    Online Security  Relevance
14                      Online Backup  Relevance
15             Device Protection Plan  Relevance
16               Premium Tech Support  Relevance
17                       Streaming TV  Relevance
18                   Streaming Movies  Relevance
19                    Streaming Music  Relevance
20                     Unlimited Data  Relevance
21                           Contract  Relevance
22                  Paperless Billing  Relevance
23                     Payment Method  Relevance
24                     Monthly Charge  Relevance
25                      Total Charges  Relevance
26                      Total Refunds  Relevance
27           Total Extra Data Charges  Relevance
28        Total Long Distance Charges  Relevance
29                      Total Revenue  Relevance

"""

""" EDA """

# Five number summary 
df.describe()
# min and max of every columns

# shape
df.shape
# There are 7043 rows and 30 columns

# columns 
df.columns

# Scatter plot 
# Here 'sns' corresponds to seaborn. 
import seaborn as sns
sns.set_style("whitegrid");
sns.FacetGrid(df,) \
   .map(plt.scatter, "Offer", "Quarter") \
   .add_legend();
plt.show();

# Notice that the blue points can be easily seperated 

sns.set_style("whitegrid");
sns.FacetGrid(df,) \
   .map(plt.scatter, "Offer", "Total Charges") \
   .add_legend();
plt.show();

# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
# Can be used when number of features are high.

sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# histplot
# histplot on Total Revenue column
sns.histplot(df['Total Revenue'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(df['Total Charges'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# box plot on all dataframe
sns.boxplot(data=df)
# There is outliers on many data columns data columns
# outliers in Total charges 

#############################################################

# There is outliers hence do outliers tratment
# Identify the duplicates
 
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series will be created

duplicate
sum(duplicate)
# Sum is 0 , there is no duplicate data
# There are many unwanted columns of data 
# hence drop that columns
df=df.drop(['Customer ID','Offer'],axis=1)

# There are some nominal type data columns 
# so use dummies for that

df.shape
# 7043 and 28 columns

df1=pd.get_dummies(df)

df1.shape
# 7043 and 48 columns

df1.columns

df1=df1.drop(['Referred a Friend_Yes','Phone Service_Yes','Multiple Lines_Yes',
          'Internet Service_Yes','Online Security_Yes','Online Backup_Yes','Device Protection Plan_Yes'
          ,'Premium Tech Support_Yes','Streaming TV_Yes','Streaming Movies_Yes','Streaming Music_Yes','Unlimited Data_Yes','Paperless Billing_Yes'],axis=1,inplace=True)

df1.shape
# Now we get 30,12
df1=df1.rename(columns={'Referred a Friend_No':'Referred a Friend',
'Phone Service_No':'Phone Service', 'Multiple Lines_No':'Multiple Lines', 'Internet Service_No':'Internet Service',
'Online Security_No':'Online Security', 'Online Backup_No':'Online Backup', 'Device Protection Plan_No':'Device Protection Plan',
'Premium Tech Support_No':'Premium Tech Support', 'Streaming TV_No':'Streaming TV', 'Streaming Movies_No':'Streaming Movies',
'Streaming Music_No':'Streaming Music', 'Unlimited Data_No':'Unlimited Data', 'Paperless Billing_No':'Paperless Billing'})

# There is scale diffrence between among the columns hence normalize it
# whenever there is mixed data apply normalization


# Identify the duplicates
 
duplicate=df1.duplicated()
# Output of this function is single columns

# if there is duplicate records output- True

# if there is no duplicate records output-False

# Series will be created

duplicate
sum(duplicate)
# Sum is 0 , there is no duplicate data

######################################################
# Treatment on outliers data

df2 = df.rename({'Monthly Charge':"Month"},axis=1)
df2.columns


IQR=df2.Month.quantile(0.75)-df.Month.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Month']
                  )
# There is scale diffrence between among the columns hence normalize it
# whenever there is mixed data apply normalization
# drop ID#
# We Know that there is scale diff. among the columns, which romove by using 
# Normalization or standasdization

#df=df.drop(['Customer ID'],axis=1)

# Apply Normalization function 

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# whenever there is mixed data apply normalization
# Now apply this normalization function to df for all the rows

df_norm=norm_fun(df2.iloc[:,1:])

# all data from is up to 1
b=df_norm.describe()
print(b)

# Ideal cluster 
# Defined the number of clusters 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans

a=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    a.append(kmeans.inertia_)

# total within sum of square




# How to select value of k from elbow curve
# when k changes from 2 to 3 , then decrease
# in a is higher than 
# when k chages from 3 to 4
# when k changes from 3 to 4.
# Whwn k value changes from 5 to 6 decreases
# in a is higher than when k chages 3 to 4 .
# When k values changes from 5 to 6 decrease
# in a is considerably less , hence considered k=3

# model Kmeans
model=KMeans(n_clusters=3)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

df2['clust']=mb

df2.head()
df2=df2.iloc[:,[7,0,1,2,3,4,5,6]]


df2.head()
df


df2.to_csv("telho.csv",encoding="utf-8")

import os

os.getcwd()

