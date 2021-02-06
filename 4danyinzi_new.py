# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:33:25 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:26:34 2020

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.style.use('seaborn')
plt.rc('font',family='Times New Roman')


data = np.loadtxt('0result.txt')
# Fixing random state for reproducibility
np.random.seed(19680801)

x_major_locator=MultipleLocator(1)#每隔几个数字显示一次

x = np.arange(1,10)
ch = data[:,0]
si = data[:,1]
dav = data[:,2]
aver = data[:,3]

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",fontsize=10,
                    ha='center', va='bottom')

labels = ['Wave height', 'Soil strength', 'Erosion','Current', 'Human',\
         'Slope', 'Liquefaction', 'Water depth','Sediment type']
labels2 = ['','Wave height', 'Soil strength', 'Erosion','Current', 'Human',\
         'Slope', 'Liquefaction', 'Water depth','Sediment type']
width = 0.2
x = np.arange(len(labels))+1
# y = [0 for _ in range(9)]

fig, ax = plt.subplots(1,1,figsize=(13,4))
# ax.plot(x,y,'b')
rects1 = ax.bar(x-width,ch, width,bottom=0.0, alpha=0.7, label='C H')
rects2 = ax.bar(x,si, width, bottom=0.0, alpha=0.7,label='S I')
rects3 = ax.bar(x+width,dav, width, bottom=0.0, alpha=0.7,label='DAV')
rects4 = ax.bar(x+2*width,aver, width, bottom=0.0, alpha=0.7,label='DAV')
ax.tick_params(labelsize=15)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xticks(x,labels2)
ax.set_xticklabels(labels2,rotation='10')
ax.set_xlabel('Influential factors',fontsize=15)
ax.set_ylim([0.6,1.4])
ax.set_ylabel('Normalized ratio',fontsize=15)
ax.legend(fontsize=15)
plt.axis('on')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.show()

# fig.savefig('para_comp.tiff',dpi=300,bbox_inches='tight')