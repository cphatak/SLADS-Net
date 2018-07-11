#! /usr/bin/env python3

import numpy as np
import os
import sys
import random
def loadSpectrum(classLabelReal,EDSData,CodePath):
    #sess = tfclf.sess
    if classLabelReal==0:
        phase = 0
        filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + EDSData.Folder +  os.path.sep + 'Phase_1' + os.path.sep + 'EDSvalid1.npy'
        vec = np.load(filename)
        randNum = random.randint(0, EDSData.NumSpectra-1);
        r = vec[randNum, :]           
        s = np.random.poisson(EDSData.Noiselambda, len(r))
    else:                
        phase = classLabelReal
        filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + EDSData.Folder +  os.path.sep + 'Phase_' + str(phase) + os.path.sep + 'EDSvalid' +str(classLabelReal) + '.npy'
        vec = np.load(filename)
        randNum = random.randint(0, EDSData.NumSpectra-1);
        r = vec[randNum, :]
        s = r + np.random.poisson(EDSData.Noiselambda, len(r))
    return s,classLabelReal

#def loadSpectrum(classLabelReal,EDSData,CodePath):
#    randFloat = random.random();
#    randNum = random.randint(0, EDSData.NumSpectra-1);
#    if classLabelReal==255: #or classLabelReal==1:
#        phase = 1
#        filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + EDSData.Folder +  os.path.sep + 'Phase_' + str(phase) + os.path.sep + 'EDSvalid1.npy'
#    else:
#        phase = 2
#        filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + EDSData.Folder +  os.path.sep + 'Phase_' + str(phase) + os.path.sep + 'EDSvalid2.npy'
#    vec = np.load(filename)
#    r = vec[randNum, :]
#    if randFloat>EDSData.ErrorSpectrumProb:       
#        if EDSData.NoiseType == 'P':
#            s = r + np.random.poisson(EDSData.Noiselambda, len(r))
#    else:
#        s = np.random.poisson(EDSData.Noiselambda, len(r))
#        phase=0.5
##    print(phase)
#    return s,phase  
  
  
#global classifySpectrum
def classifySpectrum(s,CodePath,EDSData, tfclf):
#    global sess
#    global W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, b_fc1, b_fc2, b_fc3, b_fco
#    global x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv

#    filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + 'NdFeB' +  os.path.sep + 'tfClfTrain' + os.path.sep
#    filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + 'NdFeB' +  os.path.sep
#    sys.path.append(filename)
#    from EDStfClfLoader import tfClassify
#    classLabel = tfClassify(s)

    sess = tfclf.sess
    W_conv1 = tfclf.W_conv1
    W_conv2 = tfclf.W_conv2
    W_fc1 = tfclf.W_fc1
    W_fc2 = tfclf.W_fc2
    W_fc3 = tfclf.W_fc3
    W_fco = tfclf.W_fco
#    b_conv1 = tfclf.b_conv1
#    b_conv2 = tfclf.b_conv2
    x_image = tfclf.x_image
    x = tfclf.x
    y_ = tfclf.y_
    h_conv1 = tfclf.h_conv1
    h_conv2 = tfclf.h_conv2
    h_pool1 = tfclf.h_pool1
    h_pool2 = tfclf.h_pool2
    size_hp = tfclf.size_hp
    h_flat = tfclf.h_flat
    h_fc1 = tfclf.h_fc1
    h_fc2 = tfclf.h_fc2
    h_fc3 = tfclf.h_fc3
    keep_prob = tfclf.keep_prob
    h_fc1_drop = tfclf.h_fc1_drop
    y_conv = tfclf.y_conv

    classLabel = np.argmax(sess.run(y_conv, feed_dict={x: s.reshape((1,2040)), y_: np.zeros((1,2)), keep_prob: 1.0}))
    
#    import _pickle as cPickle
#    import cPickle
#    with open(filename + 'Classifier.pkl', 'rb') as fid:
#        clf = cPickle.load(fid)
#    classLabel = clf.predict(s)
#    print(classLabel)
    return classLabel+1
    
    
    
    
def regressSpectrum(s,CodePath,EDSData):
#    filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + 'NdFeB' +  os.path.sep + 'tfRegTrain' + os.path.sep
    filename = CodePath + 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + EDSData.Folder +  os.path.sep + 'skRegTrain' + os.path.sep
    #filename = '/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/skRegTrain/'
#    sys.path.append(filename)
#    from EDStfRegLoader import tfRegress
#    regressValue = tfRegress(s)
    
#    import _pickle as cPickle
    import cPickle
    with open(filename + 'Regressor.pkl', 'rb') as fid:
        reg = cPickle.load(fid)
    regressValue = reg.predict(s)
    return regressValue

