# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:17:00 2023

@author: Vaibhav Bhorkade

"""
import pandas as pd
import matplotlib.pyplot as plt

Univ1=pd.read_excel("C:/datasets/University_Clustering.xlsx")
# We have column 'State' which is not useful drop it
Univ=Univ1.drop(['State'],axis=1)
# We Know that there is scale diff. among the columns, which romove by using 
# Normalization or standasdization
# whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to Univ dataframe for all the rows
# and columns from 1 until and Since 0 th column has university name hence
# skipped

df_norm=norm_func(Univ.iloc[:,1:]) 
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
Univ['clust']=cluster_labels
# we want to restore the column 7 to 0 th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
# now check the Univ1 dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
# from the output cluster 2 has got highest Top10
# lowest accept ratio , best faculty ratio and highest expenses
# highest graduate 

Univ1.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()

#############################################################################


