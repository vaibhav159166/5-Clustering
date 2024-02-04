# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:05:53 2023

@author: Vaibhav Bhorkade

Assignment :  Insurance Dataset.csv 

"""

"""
⦁	Analyze the information given in the following ‘Insurance
 Policy dataset’ to create clusters of persons falling in the
 same type. Refer to Insurance Dataset.csv


"""
"""
Business Objective
Minimize : Claim made and premiums
Maximaze : Days of Renew

Business constraints  
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv("C:/datasets/Insurance Dataset.csv")
df


"""
Data Dictionary

  Name_of_feature    Discription                     Type  Relevance
0   Premiums Paid  Premiums Paid             Quantitative  Relevance
1             Age            Age             Quantitative  Relevance
2   Days to Renew  Days to Renew             Quantitative  Relevance
3     Claims made    Claims made  Quantitative/Contineous  Relevance
4          Income         Income               Contineous  Relevance


"""
""" EDA """

# Five number summary
df.describe()
# min , Q1, Q2, Q3, max

print(df.shape)
# 100 rows and 5 columns

df.columns
# 5 columns name -
'''['Premiums Paid', 'Age', 'Days to Renew', 'Claims made', 
'Income']'''

# histplot

sns.histplot(df['Premiums Paid'],kde=True)
# data is right-skew and the not normallly distributed
# It is right - skew not symmetric

sns.histplot(df['Age'],kde=True)
# data is right-skew and the not normallly distributed
# not symmetric

sns.histplot(df,kde=True)
# days of Renew has high values compratable to other
#The data is showing the skewness 
# most of the right skiwed data

# BOxplot- for uderstanding outliers 
# box plot on column
sns.boxplot(df.Age)
# There is no outliers in age columns

sns.boxplot(df.Income)
# There is no outliers

# box plot on all dataframe
sns.boxplot(data=df)
# There is outliers on many data columns data columns
# Now we found outliers on Premiumns_paid, Claims made

# For Nan value
df['Income'][3]=np.NaN
df.Income.mean()

# fill nan value with the mean value 
df=df.fillna(df.Income.mean())
df

# mean
df.mean()
'''
The median of every column is as follows : 
    
Premiums Paid     12542.250000
Age                  46.110000
Days to Renew       120.400000
Claims made       12578.993367
Income           102954.545455            

Mean of income and Premiums Paid is high , lets find a median
'''

# median
df.median()
'''
The median of every column is as follows : 
    
Premiums Paid     11825.000000
Age                  45.000000
Days to Renew        89.000000
Claims made        8386.043907
Income           102977.272727

The Premiums Paid and Income median is high 
lets try for standard deviation

'''

# Standard deviation
df.std()
'''
The standard deviation of every column is as follows : 

Premiums Paid     6790.731666
Age                 13.887641
Days to Renew       88.055767
Claims made      13695.906762
Income           42943.120174

we found that standard deviation of of Income and Premiums Paid is high
'''
# Assault have high value data is not normalize

# Identify the duplicates
 
duplicate=df.duplicated()
# Output of this function is single columns

# if there is duplicate records output- True

# if there is no duplicate records output-False

# Series will be created

duplicate
sum(duplicate)
# Sum is 0 , there is no duplicate data

######################################################
# Treatment on outliers data
IQR=df.Income.quantile(0.75)-df.Income.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
                  
IQR
# but if we try as I,Iqr, or iqr then it is showing

lower_limit=df.Income.quantile(0.25)-1.5*IQR

upper_limit=df.Income.quantile(0.75)+1.5*IQR

# so make it is 0

# Replacement technique ----> masking
# Drowback of trimming technique is we are losing the data

df.describe()

'''
df=pd.DataFrame(np.where(df.Income>upper_limit,upper_limit,np.where(df.Income<lower_limit,lower_limit,df.Income)))
# if the values greater than Upper_limit
# map it to upper limit and less than lower limit
# map it to lower limit , if it is within the range
# then keep as it is

sns.boxplot(df[0])
# No outliers
'''
df=df.rename({'Premiums Paid':'Premiums_paid'},axis=1)
df['Premiums_paid']
######################################################
# OR -
# Treatment on outliers data
IQR=df.Premiumns_paid.quantile(0.75)-df.Premiumns_paid.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Premiums_paid']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

df_t=winsor.fit_transform(df[['Premiums_paid']])

# Check prious with outliers
sns.boxplot(df[['Premiums_paid']])

# Check the data columns without outliers
sns.boxplot(df_t['Premiums_paid'])
# Outliers are removed

# Now treatement for skiwed data to make normalize
# We Know that there is scale diff. among the columns, which romove by using 
# Normalization or standasdization

e=pd.get_dummies(df,drop_first=True)

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

#########################################################################
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
when k changes from 3 to 4 , then decrease
in a is higher than 
when k chages from 4 to 5
when k changes from 4 to 5.
Whwn k value changes from 5 to 6 decreases
in a is higher than when k chages 4 to 5 .
When k values changes from 5 to 6 decrease
in a is considerably less , hence considered k=4
'''

model=KMeans(n_clusters=4)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

df['clust']=mb

df.head()
df=df.iloc[:,[7,0,1,2,3,4,5,6]]
df

# Now save the Data File into folder
df.to_csv("Insurance.csv",encoding="utf-8")
# Import os
import os

os.getcwd()


