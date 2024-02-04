# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:31:50 2023

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
Minimize :To minimize Claim Time of policies and premium
Maximize :To maximize Customer Lifetime Value

Business constraints  
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Auto=pd.read_csv("C:/datasets/AutoInsurance (1).csv")

""" EDA """

# Five number summary
Auto.describe()
# min , Q1, Q2, Q3, max

print(Auto.shape)
# 9134 rows and 24 columns

Auto.columns
# 24 columns name -

# histplot

sns.histplot(Auto['Income'],kde=True)
# data is right-skew and the not normallly distributed
# It is right - skew not symmetric

sns.histplot(Auto['Number of Policies'],kde=True)
# data is right-skew and the not normallly distributed
# not symmetric

sns.histplot(Auto,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# BOxplot- for uderstanding outliers 
# box plot on column
sns.boxplot(Auto.Income)
# There is no outliers

# box plot on all dataframe
sns.boxplot(data=Auto)
# There is outliers on many data columns data columns

# Scatterplot on dataframe
sns.scatterplot(data=Auto)

# FOr Nan value
Auto['Income'][3]=np.NaN
Auto.Income.mean()

# fill nan value with the mean value 
Auto=Auto.fillna(Auto.income.mean())
Auto

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

# Identify the duplicates
 
duplicate=Auto.duplicated()
# Output of this function is single columns

# if there is duplicate records output- True

# if there is no duplicate records output-False

# Series will be created

duplicate
sum(duplicate)
# Sum is 0 , there is no duplicate data

######################################################
# Treatment on outliers data
IQR=Auto.Income.quantile(0.75)-Auto.Income.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
                  
IQR
# but if we try as I,Iqr, or iqr then it is showing

lower_limit=Auto.Income.quantile(0.25)-1.5*IQR

upper_limit=Auto.Income.quantile(0.75)+1.5*IQR

# so make it is 0

# Replacement technique ----> masking
# Drowback of trimming technique is we are losing the data

Auto.describe()


Auto=pd.DataFrame(np.where(Auto.Income>upper_limit,upper_limit,np.where(Auto.Income<lower_limit,lower_limit,Auto.Income)))
# if the values greater than Upper_limit
# map it to upper limit and less than lower limit
# map it to lower limit , if it is within the range
# then keep as it is

sns.boxplot(Auto[0])

######################################################
# OR -
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
when k changes from 2 to 3 , then decrease
in a is higher than 
when k chages from 3 to 4
when k changes from 3 to 4.
Whwn k value changes from 5 to 6 decreases
in a is higher than when k chages 3 to 4 .
When k values changes from 5 to 6 decrease
in a is considerably less , hence considered k=4
'''

model=KMeans(n_clusters=4)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

Auto['clust']=mb

Auto.head()
Auto=Auto.iloc[:,[7,0,1,2,3,4,5,6]]
Auto


Auto.to_csv("AutoInsurance.csv",encoding="utf-8")

import os

os.getcwd()


