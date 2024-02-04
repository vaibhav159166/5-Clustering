# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:23:10 2023

@author: Vaibhav Bhorkade

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans

# Let us try to understand first how k means works for two dimensional data
# for that, generate random number in the range 0 to 1
# and wth uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
# create a empty dataframe with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=["X","Y"])
# assign the values of X and Y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)
'''
with data X and Y , apply Kmeans model 
generate scatter plot
with scale / font=10

cmap=plt.cm.coolwarm:cool color cobination
'''
model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

Univ1=pd.read_excel("C:/datasets/University_Clustering.xlsx")

Univ=Univ1.drop(['State'],axis=1)
# We Know that there is scale diff. among the columns, which romove by using 
# Normalization or standasdization
# whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to Univ for all the rows

df_norm=norm_func(Univ.iloc[:,1:])

'''
What will be the ideal cluster number , will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)# total within sum of square


TWSS
# As k value increases the TWSS the TWSS value decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_ss")
'''
How to select value of k from elbow curve
when k changes from 2 to 3 , then decrease
in twss is higher than 
when k chages from 3 to 4
when k changes from 3 to 4.
Whwn k value changes from 5 to 6 decreases
in twsss is higher than when k chages 3 to 4 .
When k values changes from 5 to 6 decrease
in twss is considerably less , hence considered k=3
'''

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)

Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()

Univ1.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()

###############################################################

