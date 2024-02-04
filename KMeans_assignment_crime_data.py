# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:48:54 2023

@author: Vaibhav Bhorkade

"""
"""
Business Objective 
Minimize : Minimize the crime in all accept
Maximaze : Security 

Business constraints : Security
"""
# Data Dictionary

'''
  Name_of_feature Discription          Type    Relevance
0        Unnamed   Unnamed: 0       Nominal  Irrelevance
1          Murder      Murder  Quantitative    Relevance
2         Assault     Assault  Quantitative    Relevance
3        UrbanPop    UrbanPop  Quantitative    Relevance
4            Rape        Rape  Quantitative    Relevance
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cr=pd.read_csv("C:/datasets/crime_data.csv")
# head() - understanding dataset
cr.head()

cr.columns
# Name of columns
''' ['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'] '''

""" EDA """

# shape
cr.shape
# There are 50 rows and 5 columns

cr.dtypes
'''
In this data types 2 float data columns

Unnamed: 0     object
Murder        float64
Assault         int64
UrbanPop        int64
Rape          float64
'''
# Convert float point into int

cr['Murder'].astype(int)
cr.dtypes
cr['Rape'].astype(int)

# Five number summary
cr.describe()
# min , Q1, Q2, Q3, max

# Five number summary
cr.describe()

# Check for null values
cr.isnull()
# False

# calculating the sum of all null values 
cr.isnull().sum()
# 0

# if any null then drop it
# df.dropna()

cr.isnull().sum()
print(cr.shape)


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

# Here 'sns' corresponds to seaborn. 
sns.set_style("whitegrid");
sns.FacetGrid(cr) \
   .map(plt.scatter, "Murder", "Assault") \
   .add_legend();
plt.show();

sns.set_style("whitegrid");
sns.FacetGrid(cr) \
   .map(plt.scatter, "UrbanPop", "Rape") \
   .add_legend();
plt.show();

# Scatterplot on column
sns.scatterplot(cr.Rape)

# Scatterplot on dataframe
sns.scatterplot(data=cr)

# Pairplot on cr
sns.set_style("whitegrid");
sns.pairplot(cr);
plt.show()

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

# If any duplicates find and drop
# Identify the duplicates
 
duplicate=cr.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False

# Series will be created

duplicate
sum(duplicate)
# Sum is 0 , there is no duplicate data

# IQR 
IQR=cr.Rape.quantile(0.75)-cr.Rape.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR

# Let apply winsorizer technique for outliers tratment on Rape columns
# winsorizer treating without loss of data

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
# In privious there are 2 -3 outliers

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


print(a)
# As k value increases the a the a value decreases
plt.plot(k,a,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_ss")
'''
How to select value of k from elbow curve
when k changes from 2 to 3 , then decrease
in a is higher than 
when k chages from 3 to 4
when k changes from 3 to 4.
Whwn k value changes from 5 to 6 decreases
in a is higher than when k chages 3 to 4 .
When k values changes from 5 to 6 decrease
in a is considerably less , hence considered k=3
'''

model=KMeans(n_clusters=3)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

# clust group

cr['clust']=mb
cr.head()
cr=cr.iloc[:,[7,0,1,2,3,4,5,6]]
cr
cr.iloc[:,2:8].groupby(cr.clust).mean()

# Save the csv file in folder

cr.to_csv("crime.csv",encoding="utf-8")

import os

os.getcwd()