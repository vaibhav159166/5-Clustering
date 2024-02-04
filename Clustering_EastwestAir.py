# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:46:57 2023

@author: Vaibhav Bhorkade 

Assignment : EastWestAirlines.xlsx

"""
"""
⦁	Perform clustering for the airlines data to obtain optimum number of 
clusters. Draw the inferences from the clusters obtained. 
Refer to EastWestAirlines.xlsx dataset.
"""
"""
Business Objective 
Minimize : charges or cost, Customer Churn and Fraud.
Maximaze : Offers, Customer Engagement, Market Share.

Business constraints : Customer Satisfaction 
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("C:/datasets/EastWestAirlines.xlsx")
df.head()
""" EDA """

df.shape
# (3999, 12) - 12 columns

df.dtypes
'''
All the data integer form no need to change data types

ID#                  int64
Balance              int64
Qual_miles           int64
cc1_miles            int64
cc2_miles            int64
cc3_miles            int64
Bonus_miles          int64
Bonus_trans          int64
Flight_miles_12mo    int64
Flight_trans_12      int64
Days_since_enroll    int64
Award?               int64

'''

df.columns
'''
Index(['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award?'],
      dtype='object')
'''
# Five number summary
df.describe()

# Check for null values
df.isnull()
# False

# calculating the sum of all null values 
df.isnull().sum()
# 0

# if any null then drop it
# df.dropna()

df.isnull().sum()

# mean 
df.mean()

#####################################################

import seaborn as sns
import matplotlib.pyplot as plt

# histplot

sns.histplot(df['Balance'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(df['Bonus_miles'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 

######################################################

# box plot on column
sns.boxplot(df.Balance)
# There is outliers

# box plot on column
sns.boxplot(df.Bonus_miles)

# box plot on all dataframe
sns.boxplot(data=df)
# There is outliers on Balance and many columns

# Scatterplot on column
sns.scatterplot(df.Balance)

# Scatterplot on column
sns.scatterplot(df.Bonus_miles)

# Scatterplot on dataframe
sns.scatterplot(data=df)

# mean
df.mean()

# median
df.median()

# Standard deviation
df.std()
''' Standard deviation of the Balance and Bonus_miles is more '''

IQR=df.Balance.quantile(0.75)-df.Balance.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Balance']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

df_t=winsor.fit_transform(df[['Balance']])

sns.boxplot(df[['Balance']])
sns.boxplot(df_t['Balance'])
# We can see the outliers are removed


#################################################

# There is scale diffrence between among the columns hence normalize it
# whenever there is mixed data apply normalization
# drop ID#
df=df.drop(['ID#'],axis=1)

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(df.iloc[:,:])

b=df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")

# sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
# dendrongram()
# applying agglomerative clustering choosing 3 as clustrers
# from dendrongram
# whatever has been displayed in dendrogram is not clustering
# It is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

# Assign this series to df Dataframe as column and name the column
df['clust']=cluster_labels
# we want to restore the column 7 to 0 th position
df.shape
df=df.iloc[:,[11,1,2,3,4,5,6,7,8,9,10]]
# now check the df dataframe
df.iloc[:,2:].groupby(df.clust).mean()
# from the output cluster 2 has got highest Top10
# lowest accept ratio , best faculty ratio and highest expenses
# highest graduate 

df.to_csv("East11.csv",encoding="utf-8")
import os
os.getcwd()
