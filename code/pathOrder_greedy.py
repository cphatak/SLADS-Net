#! /usr/bin/env python3

import sys
sys.path.append('code')
import numpy as np
from scipy.io import savemat
from skimage import filters
import pylab




def pathOrder(unvisit, measured_value, x_init):

    x_len = unvisit.shape[0]
    #path = np.zeros(shape=(x_len, 2), dtype=int)
    value = np.copy(measured_value)
    path = np.copy(unvisit)
    #path_idx = np.zeros(shape=(x_len, 1), dtype=int)
    dist = 0    
    
    #print 'path=', path
    #print 'unvisit=', unvisit
    #print 'value=', value 
    
    temp_arr = np.full((x_len, 1), np.inf)
    tmp = np.zeros((1,2))
    
#    print(x_init)
#    print(path)
    
    for i in range(0, x_len):
#        temp_arr[i] = np.sqrt( (x_init[0] - path[i,0])**2 + (x_init[1] - path[i,1])**2 )
        temp_arr[i] = np.abs(x_init[0] - path[i,0]) + np.abs(x_init[1] - path[i,1])
    #idx = np.argmin(temp_arr)
    idx = np.argmin(dist + temp_arr)
    dist = dist + np.min(temp_arr)
    ###
    tmp = np.copy(path[idx])
    #print path[0], path[idx], tmp
    path[idx] = np.copy(path[0])
    path[0] = np.copy(tmp)
    #print path[0], path[idx], tmp
    
    v_tmp = np.copy(value[idx])
    value[idx] = np.copy(value[0])
    value[0] = np.copy(v_tmp)
    ###
    
    for i in range(0, x_len-1):
        temp_arr = np.full((x_len, 1), np.inf)
        for j in range(i+1, x_len):
#            temp_arr[j] = np.sqrt( (path[i,0] - path[j,0])**2 + (path[i,1] - path[j,1])**2 )
            temp_arr[j] = np.abs(path[i,0] - path[j,0]) + np.abs(path[i,1] - path[j,1])
        i#dx = np.argmin(temp_arr)
        idx = np.argmin(dist + temp_arr)
        dist = dist + np.min(temp_arr)
        
        tmp = np.copy(path[idx])
        path[idx] = np.copy(path[i+1])
        path[i+1] = np.copy(tmp)
        
        v_tmp = np.copy(value[idx])
        value[idx] = np.copy(value[i+1])
        value[i+1] = np.copy(v_tmp)
        
    
    
    return path, value, dist








