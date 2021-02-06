# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:09:40 2020

@author: admin
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = np.loadtxt('data.txt',dtype=np.float32)

from sklearn.metrics import calinski_harabasz_score,\
    davies_bouldin_score,silhouette_score
cal_score = []
sil_score = []
dav_score = []
#%%
# #计算距离的方法，可以是 “euclidean”（即 “l2”，欧氏距离），\
# #“manhattan”（即 “l1”，曼哈顿距离，有利于稀疏特征或稀疏噪声，例如文本挖掘中使用\
# #稀有词的出现作为特征时，会出现许多 0）, “cosine”（余弦距离）, ‘precomputed’
# #（预先计算的 affinity matrix）
affinity='euclidean'

for i in np.arange(2,10,1):
    pred_y = AgglomerativeClustering(n_clusters=i, affinity=affinity).fit(X)
    score1 = calinski_harabasz_score(X,pred_y.labels_)
    score2 = silhouette_score(X,pred_y.labels_)
    score3 = davies_bouldin_score(X,pred_y.labels_)
    
    cal_score.append(score1)
    sil_score.append(score2)
    dav_score.append(score3)

import matplotlib.pyplot as plt
i = np.arange(2,10,1)
fig, axs = plt.subplots(1,3,figsize=(16,4))
axs[0].plot(i,cal_score,marker='o',color='orange')
axs[0].set_xlabel('cluster')
axs[0].set_ylabel('calinski_harabasz index')

axs[1].plot(i,sil_score,marker='o',color='orange')
axs[1].set_xlabel('cluster')
axs[1].set_ylabel('silhouette index')

axs[2].plot(i,dav_score,marker='o',color='orange')
axs[2].set_xlabel('cluster')
axs[2].set_ylabel('davies_bouldin index')


plt.show()
fig.savefig('hier_n.tiff',dpi=300,bbox_inches='tight')
#%%
#%%
#计算距离的方法，可以是 “euclidean”（即 “l2”，欧氏距离），\
#“manhattan”（即 “l1”，曼哈顿距离，有利于稀疏特征或稀疏噪声，例如文本挖掘中使用\
#稀有词的出现作为特征时，会出现许多 0）, “cosine”（余弦距离）, ‘precomputed’
#（预先计算的 affinity matrix）

# pred_y = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward').fit(X)
# score1 = calinski_harabasz_score(X,pred_y.labels_)
# score2 = silhouette_score(X,pred_y.labels_)
# score3 = davies_bouldin_score(X,pred_y.labels_)
# cal_score.append(score1)
# sil_score.append(score2)
# dav_score.append(score3)

# pred_y = AgglomerativeClustering(n_clusters=3, affinity='manhattan',linkage='average').fit(X)
# score1 = calinski_harabasz_score(X,pred_y.labels_)
# score2 = silhouette_score(X,pred_y.labels_)
# score3 = davies_bouldin_score(X,pred_y.labels_)
# cal_score.append(score1)
# sil_score.append(score2)
# dav_score.append(score3)

# pred_y = AgglomerativeClustering(n_clusters=3, affinity='cosine',linkage='average').fit(X)
# score1 = calinski_harabasz_score(X,pred_y.labels_)
# score2 = silhouette_score(X,pred_y.labels_)
# score3 = davies_bouldin_score(X,pred_y.labels_)
# cal_score.append(score1)
# sil_score.append(score2)
# dav_score.append(score3)

# import matplotlib.pyplot as plt
# xx = ['euclidean','manhattan','cosine']
# # xx = np.arange(1,4)
# fig, axs = plt.subplots(1,3,figsize=(16,4))
# axs[0].plot(xx,cal_score,marker='o',color='orange')
# # axs[0].set_xticks({'euclidean','manhattan','cosine'})
# axs[0].set_xlabel('affinity')
# axs[0].set_ylabel('calinski_harabasz index')

# axs[1].plot(xx,sil_score,marker='o',color='orange')
# # axs[1].set_xticks({'euclidean','manhattan','cosine'})
# axs[1].set_xlabel('affinity')
# axs[1].set_ylabel('calinski_harabasz index')

# axs[2].plot(xx,dav_score,marker='o',color='orange')
# # axs[2].set_xticks({'euclidean','manhattan','cosine'})
# axs[2].set_xlabel('affinity')
# axs[2].set_ylabel('calinski_harabasz index')


# plt.show()
# fig.savefig('hier_n.tiff',dpi=300,bbox_inches='tight')

