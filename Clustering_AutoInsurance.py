# -*- coding: utf-8 -*-

"""
Created on Thu Oct 12 15:17:00 2023

@author: Vaibhav Bhorkade

Assignment :  Autoinsurance.csv 

"""

"""
⦁	Perform clustering on mixed data. Convert the categorical 
variables to numeric by using dummies or label encoding and perform 
normalization techniques. The data set consists of details of
 customers related to their auto insurance. Refer to
 Autoinsurance.csv dataset.


"""
"""
Business Objective
Minimize : Minimize the Claim Time
Maximaze : Maximize the Customer Lifetime Value

Business constraints  
"""
import pandas as pd
import matplotlib.pyplot as plt

Auto=pd.read_csv("C:/datasets/AutoInsurance (1).csv")

# EDA

# Five number summary
Auto.describe()
# min , Q1, Q2, Q3, max

print(Auto.shape)
# 50 rows and 5 columns

Auto.columns
# 5 columns name -

import seaborn as sns
import matplotlib.pyplot as plt

# histplot

sns.histplot(Auto['Income'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(Auto['Number of Policies'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(Auto,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# box plot on column
sns.boxplot(Auto.Income)
# There is no outliers

# box plot on all dataframe
sns.boxplot(data=Auto)
# There is outliers on many data columns data columns


# Scatterplot on column
sns.scatterplot(Auto.Response)

# Scatterplot on column
sns.scatterplot(Auto.Gender)

# Scatterplot on dataframe
sns.scatterplot(data=Auto)

# mean
Auto.mean()
'''
The median of every column is as follows : 

Customer Lifetime Value           8004.940475
Income                           37657.380009
Monthly Premium Auto                93.219291
Months Since Last Claim             15.097000
Months Since Policy Inception       48.064594
Number of Open Complaints            0.384388
Number of Policies                   2.966170
Total Claim Amount                 434.088794

Mean of income is high , lets find a median
'''

# median
Auto.median()
'''
The median of every column is as follows : 
Customer Lifetime Value           5780.182197
Income                           33889.500000
Monthly Premium Auto                83.000000
Months Since Last Claim             14.000000
Months Since Policy Inception       48.000000
Number of Open Complaints            0.000000
Number of Policies                   2.000000
Total Claim Amount                 383.945434
'''

# Standard deviation
Auto.std()
'''
The standard deviation of every column is as follows : 

Customer Lifetime Value           6870.967608
Income                           30379.904734
Monthly Premium Auto                34.407967
Months Since Last Claim             10.073257
Months Since Policy Inception       27.905991
Number of Open Complaints            0.910384
Number of Policies                   2.390182
Total Claim Amount                 290.500092

we found that standard deviation of of Income and CLV is high
'''
# Assault have high value data is not normalize

# Treatment on outliers data
IQR=Auto.Income.quantile(0.75)-Auto.Income.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Income']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

df_t=winsor.fit_transform(Auto[['Income']])

# Check prious with outliers
sns.boxplot(Auto[['Income']])

# Check the data columns without outliers
sns.boxplot(df_t['Income'])
# Outliers are removed

# Now treatement for skiwed data to make normalize

# We have column 'Customer','state','Responce' and more which is not useful drop it
Auto=Auto.drop(['State','Customer','Effective To Date'],axis=1)
# We Know that there is scale diff. among the columns, which romove by using 
# Normalization or standasdization
e=pd.get_dummies(Auto,drop_first=True)
# Normalization function written where ethnic arguments is passed
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(e)
b=df_norm.describe()
# whenever there is mixed data apply normalization

# Now apply this normalization function to Univ dataframe for all the rows
# and columns from 1 until and Since 0 th column has university name hence
# skipped

df_norm=norm_func(e.iloc[:,:]) 
# you can check the df_norm dataframe which is scaled between values from 0 and 1
b=df_norm.describe()
# Before you can apply clustering , you need to plot dendrogram first
# now to create dendrogram , we need to measure distance,
# we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
# Linkage function gives us hierarchical or aglomerative clustering
# ref the help for linkage
z=linkage(df_norm,method="complete",metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
# ref help of dendrogram
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

# Assign this series to Univ Dataframe as column and name the column
e['clust']=cluster_labels
e.shape

# now check the Univ1 dataframe
e.iloc[:,2:].groupby(e.clust).mean()
# from the output cluster 2 has got highest Top10
# lowest accept ratio , best faculty ratio and highest expenses
# highest graduate 

e.to_csv("AutoInsurance1.csv",encoding="utf-8")
import os
os.getcwd()

