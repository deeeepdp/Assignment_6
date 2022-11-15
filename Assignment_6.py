#!/usr/bin/env python
# coding: utf-8

# QUESTION 2:
# Use CC_GENERAL.csv given in the folder and apply:
# a) Preprocess the data by removing the categorical column and filling the missing values.
# b) Apply StandardScaler() and normalize() functions to scale and normalize raw input data.
# c) Use PCA with K=2 to reduce the input dimensions to two features.
# d) Apply Agglomerative Clustering with k=2,3,4 and 5 on reduced features and visualize result for each k value using scatter plot.
# e) Evaluate different variations using Silhouette Scores and Visualize results with a bar chart

# In[76]:


import pandas as pd
import numpy as np

from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

#add all lib here 


# In[77]:


dataset= pd.read_csv("C:/Users/deepp/OneDrive/Desktop/ML/Assignment_6/CC_GENERAL.CSV") #add data with our machine 


# In[78]:


dataset #only print dataset


# In[79]:


dataset.isnull().sum() #check dartaset is null or not null with isnull fun.


# In[80]:


dataset = dataset.drop("CUST_ID", axis=1) #here we drop cat. data like cust_id


# In[81]:


#checking for Null values
dataset.isnull().sum()


# In[82]:


dataset = dataset.fillna(dataset.mean()) #filling null values with mean values


# In[83]:


dataset.isnull().sum() #check again our data is full fill their req..


# In[84]:


scaler = StandardScaler()
X_Scale = scaler.fit_transform(dataset)
print(X_Scale) #apply StandardScaler()


# In[85]:


d = preprocessing.normalize(X_Scale)
scaled_df = pd.DataFrame(d)
scaled_df.head() #apply normalize() functions


# In[86]:


print(scaled_df) 


# In[87]:


pca=PCA(2)
x_pca = pca.fit_transform(d)
x_pca = pd.DataFrame(x_pca)
x_pca.columns = ['P1','P2'] #Use PCA with K=2 to reduce the input dimensions to two features


# In[88]:


x_pca.head() #head function 


# In[ ]:


#Apply Agglomerative Clustering with k=2,3,4 and 5 on reduced features and visualize result for each k value using scatter plot.


# In[89]:


from sklearn.cluster import AgglomerativeClustering 
AC2 = AgglomerativeClustering(2)
a = AC2.fit_predict(x_pca) #Agglomerative Clustering with k=2


# In[90]:


plt.scatter(x_pca['P1'],x_pca['P2'],c=a ,cmap='viridis')
plt.show() #plt.show use for pictorial graph presentation 


# In[102]:


AC3 = AgglomerativeClustering(3)
a = AC3.fit_predict(x_pca)
plt.scatter(x_pca['P1'],x_pca['P2'],c=a ,cmap='viridis')
plt.show()  #Agglomerative Clustering with k=3 and print


# In[103]:


AC4 = AgglomerativeClustering(4)
a = AC4.fit_predict(x_pca)
plt.scatter(x_pca['P1'],x_pca['P2'],c=a ,cmap='plasma')
plt.show()  #Agglomerative Clustering with k=4 and print


# In[104]:


AC5 = AgglomerativeClustering(5)
a = AC5.fit_predict(x_pca)
plt.scatter(x_pca['P1'],x_pca['P2'],c=a ,cmap='magma')
plt.show()  #Agglomerative Clustering with k=5 and print 


# In[110]:



k = [2, 3, 4, 5]

# Appending the silhouette scores of the different models to the list
sss = []
sss.append(silhouette_score(x_pca, AC2.fit_predict(x_pca)))
sss.append(silhouette_score(x_pca, AC3.fit_predict(x_pca)))
sss.append(silhouette_score(x_pca, AC4.fit_predict(x_pca)))
sss.append(silhouette_score(x_pca, AC5.fit_predict(x_pca)))



# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()
#reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html


# In[ ]:




