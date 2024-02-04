# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:13:07 2023

@author: Vaibhav Bhorkade

Assignment Crime data
"""

"""
⦁	Perform clustering for the crime data and identify the number of 
clusters formed and draw inferences. Refer to crime_data.csv dataset.

"""
"""
Business Objective 
Minimize : crime
Maximaze : Security

Business constraints : Security
"""
import pandas as pd
import matplotlib.pyplot as plt

cr=pd.read_csv("C:/datasets/crime_data.csv")
# head - understanding dataset
cr.head()

# EDA

# Five number summary
cr.describe()
# min , Q1, Q2, Q3, max

print(cr.shape)
# 50 rows and 5 columns

cr.columns
# 5 columns name -'Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'

import seaborn as sns
import matplotlib.pyplot as plt

# histplot

sns.histplot(cr['Murder'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(cr['Assault'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(cr,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# box plot on column
sns.boxplot(cr.Murder)
# There is no outliers

# box plot on column
sns.boxplot(cr.Assault)
# There is no outliers

# box plot on all dataframe
sns.boxplot(data=cr)
# There is outliers on Rap data columns


# Scatterplot on column
sns.scatterplot(cr.Murder)

# Scatterplot on column
sns.scatterplot(cr.Rape)

# Scatterplot on dataframe
sns.scatterplot(data=cr)

# mean
cr.mean()
# Murder        7.788
# Assault     170.760
# UrbanPop     65.540
# Rape         21.232
''' Mean of the Assault column is high '''

# median
cr.median()
#Murder        7.25
#Assault     159.00
#UrbanPop     66.00
#Rape         20.10
''' Median of Assault column is high'''

# Standard deviation
cr.std()
'''
Murder       4.355510
Assault     83.337661
UrbanPop    14.474763
Rape         9.366385
'''
# Assault have high value data is not normalize

# Treatment on outliers data

# Only Rape column has outliers hence do outliers tratment on it

IQR=cr.Rape.quantile(0.75)-cr.Rape.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR

# Let apply winsorizer technique for outliers tratment on Rape columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Rape']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

df_t=winsor.fit_transform(cr[['Rape']])

# Check prious with outliers
sns.boxplot(cr[['Rape']])

# Check the data columns without outliers
sns.boxplot(df_t['Rape'])
# Outliers are removed

# Now treatement for skiwed data to make normalize

######################################################################

# There is scale diffrence between among the columns hence normalize it
cr.columns
# whenever there is mixed data apply normalization
# Unnamed: 0 is unwanted data column hence drop it 
cr=cr.drop(['Unnamed: 0'],axis=1)
# column deleted
cr.head()

# Normalize the data using norm function

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# Apply the norm_fun to data 
df_norm=norm_fun(cr.iloc[:,:])

info=df_norm.describe()

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
cr['clust']=cluster_labels
# we want to restore the column 7 to 0 th position
cr.shape
# (50, 5)
cr=cr.iloc[:,[4,1,2,3]]
# now check the df dataframe
cr.iloc[:,2:].groupby(cr.clust).mean()
# from the output cluster 2 has got highest Top10
# lowest accept ratio , best faculty ratio and highest expenses
# highest graduate 

cr.to_csv("Crime.csv",encoding="utf-8")
import os
os.getcwd()
