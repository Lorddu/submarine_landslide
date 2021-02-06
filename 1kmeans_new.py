# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:08:30 2020

@author: admin
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.loadtxt('data.txt',dtype=np.float32)

from sklearn.metrics import calinski_harabasz_score,\
    davies_bouldin_score,silhouette_score
cal_score = []
sil_score = []
dav_score = []
for i in range(2,11):
    ##构建并训练模型
    kmeans = KMeans(n_clusters = i,random_state=123).fit(X)
    score1 = calinski_harabasz_score(X,kmeans.labels_)
    score3 = davies_bouldin_score(X,kmeans.labels_)
    score2 = silhouette_score(X,kmeans.labels_)
    
    cal_score.append(score1)
    dav_score.append(score3)
    sil_score.append(score2)
    # print('iris数据聚%d类calinski_harabasz指数为：%f'%(i,score))
# plot results
i = np.arange(2,11,1)
fig, axs = plt.subplots(1,3,figsize=(15,4))
axs[0].plot(i,cal_score,marker='o',color='lightcoral')
axs[0].set_xlabel('cluster')
axs[0].set_ylabel('calinski_harabasz index')

axs[1].plot(i,sil_score,marker='o',color='lightcoral')
axs[1].set_xlabel('cluster')
axs[1].set_ylabel('silhouette index')

axs[2].plot(i,dav_score,marker='o',color='lightcoral')
axs[2].set_xlabel('cluster')
axs[2].set_ylabel('davies_bouldin index')

plt.show()
# fig.savefig('kmeans.tiff',dpi=300,bbox_inches='tight')



