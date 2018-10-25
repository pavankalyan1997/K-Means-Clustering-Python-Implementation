#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from collections import defaultdict
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


Dataset=pd.read_csv('customer_new1.csv')
m=1000
x1=Dataset.iloc[:,2:3].values
x2=Dataset.iloc[:,3:4].values
x3=Dataset.iloc[:,4:5].values
X=np.c_[x1,x2,x3]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1,x2,x3,marker='.',c='black')
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('No of Transactions')
#ax.set_title('Input dataset for performing clustering')
plt.show()

"""
Steps involved in K Means Clustering
1. Initialize two examples of the training data set as Centroids
2. Loop over the num of iterations to perform the clustering
2.a. For each training example compute the euclidian distance from the centroid and assign the cluster
based on the minimal distance.
2.b Adjust the centroid of each cluster by taking the average of all the training examples which belonged 
to that cluster on the basis of the computations performed in step 3.a

"""
#Step 1.initialize number of clusters
K=3
Centroids=np.array([]).reshape(3,0)

for i in range(K):
    rand=rd.randint(0,m)
    Centroids=np.c_[Centroids,X[rand]]

#Plot the data set with initial centroids
"""
i=0
plt.scatter(x1,x2,marker='.',c='black')
for i in range(K):
    plt.scatter(Centroids[:,i][0],Centroids[:,i][1],marker='x')
plt.xlabel('x1(first input feature)')
plt.ylabel('x2(second input feature)')
plt.title('Input dataset for performing clustering with initial Centroids')
plt.show()
"""

ax1 = plt.axes(projection='3d')
ax1.scatter(x1,x2,x3,marker='.',c='black')
for i in range(K):
    ax1.scatter(Centroids[:,i][0],Centroids[:,i][1],Centroids[:,i][2],marker='x')
ax1.set_xlabel('Age')
ax1.set_ylabel('Income')
ax1.set_zlabel('No of Transactions')
#ax.title('Input dataset for performing clustering')
plt.show()



#step2
num_iter=100
Output=defaultdict()
for n in range(num_iter):
    #step 2.a
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(K):
        tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        
    C=np.argmin(EuclidianDistance,axis=1)+1
    #step 2.b
    Y=defaultdict()
    for k in range(K):
        Y[k+1]=np.array([]).reshape(3,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X[i]]
        
    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=1)
        
    Output=Y

ax2= plt.axes(projection='3d')
for k in range(K):
    ax2.scatter(Output[k+1][0,:],Output[k+1][1,:],Output[k+1][2,:],marker='.')
    ax2.scatter(Centroids[:,k][0],Centroids[:,k][1],Centroids[:,k][2],marker='x')
ax2.set_xlabel('Age')
ax2.set_ylabel('Income')
ax2.set_zlabel('No of Transactions')
plt.show()

for k in range(K):
    plt.scatter(Output[k+1][1,:],Output[k+1][2,:],marker='.')
    plt.scatter(Centroids[:,k][1],Centroids[:,k][2],marker='x')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.show()



    
    

    