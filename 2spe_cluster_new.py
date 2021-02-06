# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:19:07 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

plt.style.use('default')

X = np.loadtxt('data.txt',dtype=np.float32)

spectral=SpectralClustering()

from sklearn.metrics import calinski_harabasz_score,\
    davies_bouldin_score,silhouette_score
cal_score = []
sil_score = []
dav_score = []
 
#默认使用的是高斯核，需要对n_cluster和gamma进行调参，选择合适的参数
# gamma = 0.01
# # for index,k in enumerate((3,4)):
# for k in np.arange(2,11,1):
#     pred_y=SpectralClustering(n_clusters=k, gamma=gamma).fit(X)
#     score1 = calinski_harabasz_score(X,pred_y.labels_)
#     score2 = silhouette_score(X,pred_y.labels_)
#     score3 = davies_bouldin_score(X,pred_y.labels_)
    
#     cal_score.append(score1)
#     sil_score.append(score2)
#     dav_score.append(score3)
# #此处i和k是一个意思
# i = np.arange(2,11,1)
# fig, axs = plt.subplots(1,3,figsize=(15,4))
# axs[0].plot(i,cal_score,marker='o')
# axs[0].set_xlabel('cluster')
# axs[0].set_ylabel('calinski_harabasz index')

# axs[1].plot(i,sil_score,marker='o')
# axs[1].set_xlabel('cluster')
# axs[1].set_ylabel('silhouette index')

# axs[2].plot(i,dav_score,marker='o')
# axs[2].set_xlabel('cluster')
# axs[2].set_ylabel('davies_bouldin index')

# plt.show()
# fig.savefig('spe_k.tiff',dpi=300,bbox_inches='tight')

k = 3
# for index,k in enumerate((3,4)):
for gamma in np.arange(0.01,5,0.1):
    pred_y=SpectralClustering(n_clusters=k, gamma=gamma).fit(X)
    score1 = calinski_harabasz_score(X,pred_y.labels_)
    score2 = silhouette_score(X,pred_y.labels_)
    score3 = davies_bouldin_score(X,pred_y.labels_)
    
    cal_score.append(score1)
    sil_score.append(score2)
    dav_score.append(score3)
#此处i和k是一个意思
i = np.arange(0.01,5,0.1)
fig, axs = plt.subplots(1,3,figsize=(16,4))
axs[0].plot(i,cal_score,'--')
axs[0].set_xlabel('gamma')
axs[0].set_ylabel('calinski_harabasz index')

axs[1].plot(i,sil_score,'--')
axs[1].set_xlabel('gamma')
axs[1].set_ylabel('silhouette index')

axs[2].plot(i,dav_score,'--')
axs[2].set_xlabel('gamma')
axs[2].set_ylabel('davies_bouldin index')

plt.show()
# # fig.savefig('spe_gamma.tiff',dpi=300,bbox_inches='tight')

