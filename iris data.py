# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:03:58 2020

@author: shell
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pandas as pd

iris=load_iris()
#experimenting with data 
features=iris.data.T

sepal_length=features[0]
sepal_width=features[1]
petal_length=features[2]
petal_width=features[3]

sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]

fig, (ax1,ax2)=plt.subplots(1,2)

#plt.scatter(sepal_length,sepal_width, c=iris.target)
#plt.xlabel(sepal_length_label)
#plt.ylabel(sepal_width_label)
sns.scatterplot(sepal_length,sepal_width,hue=iris.target,ax=ax1)


#actual code split in 75 and 25
X_train,X_test,y_train,y_test= train_test_split(iris['data'],iris['target'])
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
X_new=np.array([[5.0,2.9,1.0,0.2]])
prediction=knn.predict(X_new)
print(prediction)
print(knn.score(X_test,y_test))


#corelation of data with target
data=pd.DataFrame(iris.data)
target=pd.DataFrame(iris.target)
df = pd.concat([data, target], axis = 1)
sns.heatmap(df.corr(), annot = True,ax=ax2)

