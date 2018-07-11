# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt

measure = np.load('MeasuredIdxs.npy')
measure_order = np.load('MeasuredIdxs_order.npy')


#print measure.shape
#print measure_order.shape


x_len = measure.shape[0]
dist = np.zeros(shape=(x_len-1, 1))
dist_order = np.zeros(shape=(x_len-1, 1))
dist_cdf = np.zeros(shape=(x_len-1, 1))
dist_order_cdf = np.zeros(shape=(x_len-1, 1))


for i in range(0, x_len-1):
    dist[i] = np.sqrt( (measure[i+1][0] - measure[i][0])**2 + (measure[i+1][1] - measure[i][1])**2 )
    dist_order[i] = np.sqrt( (measure_order[i+1][0] - measure_order[i][0])**2 + (measure_order[i+1][1] - measure_order[i][1])**2 )

dist_cdf[0] = dist[0]
dist_order_cdf[0] = dist_order[0]
for i in range(1, x_len-1):
    dist_cdf[i] = dist_cdf[i-1] + dist[i]
    dist_order_cdf[i] = dist_order_cdf[i-1] + dist_order[i]


sum_path = sum(dist)
sum_path_order = sum(dist_order)

#print sum_path
#print sum_path_order

'''
plt.plot(dist, hold='True')
plt.plot(dist_order)
plt.show()
'''

plt.figure()
plt.plot(dist_cdf, hold='True', label = 'original path')
plt.plot(dist_order_cdf, label = 'shortest path')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(0.32, 0.85), loc=4, borderaxespad=0.)
plt.title('cumulative travel distance')
plt.xlabel('pixel')
plt.ylabel('traved distance')
plt.show()

print dist_cdf[-1]
print dist_order_cdf[-1]





####

TD = np.load('TD.npy')
TD_order1 = np.load('TD_order1.npy')
TD_order2 = np.load('TD_order2.npy')
dist_arr = np.load('dist_arr.npy')
dist_arr_order1 = np.load('dist_arr_order1.npy')
dist_arr_order2 = np.load('dist_arr_order2.npy')

plt.figure()
plt.plot(dist_arr, TD, hold='True', label = 'original path', marker = '', lineWidth='2')
plt.plot(dist_arr_order1, TD_order1, label = 'shortest - Manhattan', marker = '', LineWidth='2')
plt.plot(dist_arr_order2, TD_order2, label = 'shortest - Euclidean', marker = '', lineWidth='2')
plt.legend(bbox_to_anchor=(0.98, 0.80), loc=4, borderaxespad=0.)
#plt.title('Total Distortion vs. travel distance')
plt.xlabel('Travel Distance (pixel)')
plt.ylabel('Total Distortion')
plt.show()




























