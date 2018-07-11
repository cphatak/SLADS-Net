#! /usr/bin/env python3

import sys
sys.path.append('code')
import numpy as np
from scipy.io import savemat
from skimage import filters
import pylab


from performMeasurements import perfromMeasurements
from performMeasurements import perfromInitialMeasurements
from updateERDandFindNewLocation import updateERDandFindNewLocationFirst
from updateERDandFindNewLocation import updateERDandFindNewLocationAfter
from computeStopCondRelated import computeStopCondFuncVal
from computeStopCondRelated import checkStopCondFuncThreshold
from performMeasurements import updateMeasurementArrays
from performReconOnce import performReconOnce
from loader import loadTestImage

from pathOrder_greedy import pathOrder



def runSLADSSimulationOnce(Mask,CodePath,ImageSet,SizeImage,StopCondParams,Theta,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,SavePath,SimulationRun,ImNum,ImageExtension,PlotResult,Classify):
  
    MeasuredIdxs = np.transpose(np.where(Mask==1))
    UnMeasuredIdxs = np.transpose(np.where(Mask==0))
    
    ContinuousMeasuredValues = perfromInitialMeasurements(CodePath,ImageSet,ImNum,ImageExtension,Mask,SimulationRun)
    
####
#    MeasuredIdxs, ContinuousMeasuredValues, travdist = pathOrder(MeasuredIdxs, ContinuousMeasuredValues, np.array([0,0]))
####
    
    
    if Classify=='2C':
        Threshold = filters.threshold_otsu(ContinuousMeasuredValues)
        print('Threhold found using the Otsu method for 2 Class classification = ' + str(Threshold))
        MeasuredValues = ContinuousMeasuredValues < Threshold
        MeasuredValues = MeasuredValues+0
#    elif Classify=='MC':
        #### Classification function to output NewValues ##################
        # NewValues is the vector of measured values post classification
    elif Classify=='N':
        MeasuredValues=ContinuousMeasuredValues
    
    # Perform SLADS
    IterNum=0
    Stop=0
    NumSamples = np.shape(MeasuredValues)[0]
    StopCondFuncVal=np.zeros(( int((SizeImage[0]*SizeImage[1])*(StopCondParams.MaxPercentage)/100)+10,2 ))
    while Stop !=1:
        
        if IterNum==0:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,BatchSamplingParams)           
        else:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect)
    
        NewContinuousValues = perfromMeasurements(NewIdxs,CodePath,ImageSet,ImNum,ImageExtension,MeasuredIdxs,BatchSamplingParams,SimulationRun)
        
####
#        NewIdxs, NewContinuousValues, travdist = pathOrder(NewIdxs, NewContinuousValues, MeasuredIdxs[-1])	
####
        
        ContinuousMeasuredValues = np.hstack((ContinuousMeasuredValues,NewContinuousValues))
        

        
        
        if Classify=='2C':           
            NewValues = NewContinuousValues > Threshold
            NewValues = NewValues+0
#        elif Classify=='MC':
            #### Classification function to output NewValues ##################
            # NewValues is the vector of measured values post classification            
        elif Classify=='N':
            NewValues=NewContinuousValues    


        Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParams)
    
        NumSamples = np.shape(MeasuredValues)[0]
    
        StopCondFuncVal=computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParams,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParams)
            
        Stop = checkStopCondFuncThreshold(StopCondParams,StopCondFuncVal,NumSamples,IterNum,SizeImage)
        if PlotResult=='Y' and np.remainder(NumSamples,round(0.01*SizeImage[0]*SizeImage[1])) ==0:
            print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled')
        IterNum += 1
        
        
        ###
#        np.save(SavePath + 'MeasuredIdxs_order', MeasuredIdxs_Order)
#        np.save(SavePath + 'ContinuousMeasuredValues_order', ContinuousMeasuredValues_order)
        ###
        
    
    np.save(SavePath + 'MeasuredValues', MeasuredValues)
    np.save(SavePath + 'MeasuredIdxs', MeasuredIdxs)
    np.save(SavePath + 'UnMeasuredIdxs', UnMeasuredIdxs)
    np.save(SavePath + 'StopCondFuncVal',StopCondFuncVal)
    np.save(SavePath + 'ContinuousMeasuredValues',ContinuousMeasuredValues)
    savemat(SavePath + 'MeasuredIdxs.mat',dict(MeasuredIdxs=MeasuredIdxs))
    savemat(SavePath + 'MeasuredValues.mat',dict(MeasuredValues=MeasuredValues))
    savemat(SavePath + 'UnMeasuredIdxs.mat',dict(UnMeasuredIdxs=UnMeasuredIdxs))
    savemat(SavePath + 'StopCondFuncVal.mat',dict(StopCondFuncVal=StopCondFuncVal))
    savemat(SavePath + 'ContinuousMeasuredValues.mat',dict(ContinuousMeasuredValues=ContinuousMeasuredValues))

    if PlotResult=='Y': 
        print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled before stopping')
        Difference,ReconImage = performReconOnce(SavePath,TrainingInfo,Resolution,SizeImage,ImageType,CodePath,ImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues)
        TD = Difference/(SizeImage[0]*SizeImage[1])
        img=loadTestImage(CodePath,ImageSet,ImNum,ImageExtension,SimulationRun)  
        print('')
        print('')
        print('######################################')
        print('Total Distortion = ' + str(TD))
        
        from plotter import plotAfterSLADSSimulation  
        plotAfterSLADSSimulation(Mask,ReconImage,img)
        pylab.show()

        
def runSLADSOnce(Mask,CodePath,SizeImage,StopCondParams,Theta,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,SavePath,SimulationRun,ImNum,PlotResult,Classify):
  
    MeasuredIdxs = np.transpose(np.where(Mask==1))
    UnMeasuredIdxs = np.transpose(np.where(Mask==0))
    
    ##################################################################
    # CODE HERE
    # Plug in Your Measurement Routine
    # Please use 'MeasuredValues' as output variable
    # ContinuousMeasuredValues = perfromMeasurements(Mask)
    ##################################################################
    
    if Classify=='2C':
        Threshold = filters.threshold_otsu(ContinuousMeasuredValues)
        print('Threhold found using the Otsu method for 2 Class classification = ' + str(Threshold))
        MeasuredValues = ContinuousMeasuredValues < Threshold
        MeasuredValues = MeasuredValues+0
#    elif Classify=='MC':
        #### Classification function to output NewValues ##################
        # NewValues is the vector of measured values post classification
    elif Classify=='N':
        MeasuredValues=ContinuousMeasuredValues
    
    # Perform SLADS
    IterNum=0
    Stop=0
    NumSamples = np.shape(MeasuredValues)[0]
    StopCondFuncVal=np.zeros(( int((SizeImage[0]*SizeImage[1])*(StopCondParams.MaxPercentage)/100)+10,2 ))
    while Stop !=1:
        
        if IterNum==0:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,BatchSamplingParams)           
        else:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect)
        
        ##################################################################
        # CODE HERE
        # Plug in Your Measurement Routine
        # Please use 'NewContValues' as output variable
        # NewContinuousValues = perfromMeasurements(NewIdxs)
        ##################################################################    
        
        ContinuousMeasuredValues = np.hstack((ContinuousMeasuredValues,NewContinuousValues))
        if Classify=='2C':           
            NewValues = NewContinuousValues > Threshold
            NewValues = NewValues+0
#        elif Classify=='MC':
            #### Classification function to output NewValues ##################
            # NewValues is the vector of measured values post classification    
        elif Classify=='N':
            NewValues=NewContinuousValues    


        Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParams)
    
        NumSamples = np.shape(MeasuredValues)[0]
    
        StopCondFuncVal=computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParams,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParams)
            
        Stop = checkStopCondFuncThreshold(StopCondParams,StopCondFuncVal,NumSamples,IterNum,SizeImage)
        if PlotResult=='Y' and np.remainder(NumSamples,round(0.01*SizeImage[0]*SizeImage[1])) ==0:
            print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled')
        IterNum += 1
        
    
    np.save(SavePath + 'MeasuredValues', MeasuredValues)
    np.save(SavePath + 'MeasuredIdxs', MeasuredIdxs)
    np.save(SavePath + 'UnMeasuredIdxs', UnMeasuredIdxs)
    np.save(SavePath + 'StopCondFuncVal',StopCondFuncVal)
    np.save(SavePath + 'ContinuousMeasuredValues',ContinuousMeasuredValues)
    savemat(SavePath + 'MeasuredIdxs.mat',dict(MeasuredIdxs=MeasuredIdxs))
    savemat(SavePath + 'MeasuredValues.mat',dict(MeasuredValues=MeasuredValues))
    savemat(SavePath + 'UnMeasuredIdxs.mat',dict(UnMeasuredIdxs=UnMeasuredIdxs))
    savemat(SavePath + 'StopCondFuncVal.mat',dict(StopCondFuncVal=StopCondFuncVal))
    savemat(SavePath + 'ContinuousMeasuredValues.mat',dict(ContinuousMeasuredValues=ContinuousMeasuredValues))
    
    if PlotResult=='Y': 
        print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled before stopping')
        
        from plotter import plotAfterSLADS  
        plotAfterSLADS(Mask,ReconImage)
        pylab.show()









def runEDSSLADSSimulationOnce(Mask,CodePath,ImageSet,SizeImage,StopCondParams,Theta,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,SavePath,SimulationRun,ImNum,ImageExtension,PlotResult,Classify,EDSData, tfclf):
    
#    global sess, new_saver1
#    global W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, b_fc1, b_fc2, b_fc3, b_fco
#    global x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv
    
    from relatedToEDS import loadSpectrum
    from relatedToEDS import classifySpectrum
    from relatedToEDS import regressSpectrum 
    
    y_tar = np.zeros(100)
    for i in range(0, 100):
        y_tar[i] = i

    MeasuredIdxs = np.transpose(np.where(Mask==1))
    UnMeasuredIdxs = np.transpose(np.where(Mask==0))
    
    ContinuousMeasuredValues = perfromInitialMeasurements(CodePath,ImageSet,ImNum,ImageExtension,Mask,SimulationRun)
    if Classify=='2C':
        Threshold = filters.threshold_otsu(ContinuousMeasuredValues)
        print('Threhold found using the Otsu method for 2 Class classification = ' + str(Threshold))
        MeasuredValues = ContinuousMeasuredValues < Threshold
        MeasuredValues = MeasuredValues+0
    #    elif Classify=='MC':
    #### Classification function to output NewValues ##################
    # NewValues is the vector of measured values post classification
    elif Classify == 'EDS':
        MeasuredValues = ContinuousMeasuredValues
        MeasuredWithoutnoiseValues = ContinuousMeasuredValues
        for t in range(0,len(ContinuousMeasuredValues)):
            s,phase = loadSpectrum(ContinuousMeasuredValues[t],EDSData,CodePath)
            regressValue=regressSpectrum(s,CodePath,EDSData)
            #print(np.var(np.abs(regressValue-y_tar)))
            if np.var(np.abs(regressValue-y_tar)) <= 100.0:
#            if(1):
                classLabel=classifySpectrum(s,CodePath,EDSData, tfclf)
                if np.int(phase) == np.int(classLabel):
                    print("true")
                else:
                    print("wrong")
#                classLabel = np.argmax(sess.run(y_conv, feed_dict={x: s, y_: np.zeros((1,2)), keep_prob: 1.0}))
            else:
                classLabel=0
            #print(classLabel)
            MeasuredValues[t]=classLabel
            MeasuredWithoutnoiseValues[t]=phase
    elif Classify=='N':
        MeasuredValues=ContinuousMeasuredValues
    
    # Perform SLADS
    IterNum=0
    Stop=0
    NumSamples = np.shape(MeasuredValues)[0]
    StopCondFuncVal=np.zeros(( int((SizeImage[0]*SizeImage[1])*(StopCondParams.MaxPercentage)/100)+10,2 ))
    while Stop !=1:
        
        if IterNum==0:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,BatchSamplingParams)
        else:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect)
        
        NewContinuousValues = perfromMeasurements(NewIdxs,CodePath,ImageSet,ImNum,ImageExtension,MeasuredIdxs,BatchSamplingParams,SimulationRun)
        ContinuousMeasuredValues = np.hstack((ContinuousMeasuredValues,NewContinuousValues))
        if Classify=='2C':
            NewValues = NewContinuousValues > Threshold
            NewValues = NewValues+0
        #        elif Classify=='MC':
        #### Classification function to output NewValues ##################
        # NewValues is the vector of measured values post classification
        elif Classify == 'EDS':
            NewValues = NewContinuousValues
            NewMeasuredWithoutnoiseValues = NewContinuousValues
            if BatchSamplingParams.NumSamplesPerIter>1:
                for t in range(0,len(NewContinuousValues)):
                    s,phase = loadSpectrum(NewContinuousValues[t],EDSData,CodePath)
                    regressValue=regressSpectrum(s,CodePath,EDSData)
                    if np.var(np.abs(regressValue-y_tar)) <= 100.0:
                    #if(1):
                        classLabel=classifySpectrum(s,CodePath,EDSData, tfclf)
                    else:
                        classLabel=0
                    NewValues[t]=classLabel
                    NewMeasuredWithoutnoiseValues[t]=phase
            else:
                s,phase = loadSpectrum(NewContinuousValues,EDSData,CodePath)
                regressValue=regressSpectrum(s,CodePath,EDSData)
                if np.var(np.abs(regressValue-y_tar)) <= 100.0:
#                if(1):
                        classLabel=classifySpectrum(s,CodePath,EDSData, tfclf)
                else:
                        classLabel=0               
                NewValues=classLabel
                NewMeasuredWithoutnoiseValues=phase
        elif Classify=='N':
            NewValues=NewContinuousValues
        MeasuredWithoutnoiseValues  = np.hstack((MeasuredWithoutnoiseValues,NewMeasuredWithoutnoiseValues))
        Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParams)
        
        NumSamples = np.shape(MeasuredValues)[0]
        
        StopCondFuncVal=computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParams,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParams)
        
        Stop = checkStopCondFuncThreshold(StopCondParams,StopCondFuncVal,NumSamples,IterNum,SizeImage)
        if PlotResult=='Y' and np.remainder(NumSamples,round(0.01*SizeImage[0]*SizeImage[1])) ==0:
            print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled')
            np.save(SavePath + 'MeasuredValues', MeasuredValues)
            np.save(SavePath + 'MeasuredIdxs', MeasuredIdxs)
            np.save(SavePath + 'UnMeasuredIdxs', UnMeasuredIdxs)
            np.save(SavePath + 'StopCondFuncVal',StopCondFuncVal)
            np.save(SavePath + 'ContinuousMeasuredValues',ContinuousMeasuredValues)
            np.save(SavePath + 'MeasuredWithoutnoiseValues',MeasuredWithoutnoiseValues)		
        IterNum += 1


    np.save(SavePath + 'MeasuredValues', MeasuredValues)
    np.save(SavePath + 'MeasuredIdxs', MeasuredIdxs)
    np.save(SavePath + 'UnMeasuredIdxs', UnMeasuredIdxs)
    np.save(SavePath + 'StopCondFuncVal',StopCondFuncVal)
    np.save(SavePath + 'ContinuousMeasuredValues',ContinuousMeasuredValues)
    np.save(SavePath + 'MeasuredWithoutnoiseValues',MeasuredWithoutnoiseValues)
    
    savemat(SavePath + 'MeasuredIdxs.mat',dict(MeasuredIdxs=MeasuredIdxs))
    savemat(SavePath + 'MeasuredValues.mat',dict(MeasuredValues=MeasuredValues))
    savemat(SavePath + 'UnMeasuredIdxs.mat',dict(UnMeasuredIdxs=UnMeasuredIdxs))
    savemat(SavePath + 'StopCondFuncVal.mat',dict(StopCondFuncVal=StopCondFuncVal))
    savemat(SavePath + 'ContinuousMeasuredValues.mat',dict(ContinuousMeasuredValues=ContinuousMeasuredValues))
    savemat(SavePath + 'MeasuredWithoutnoiseValues.mat',dict(MeasuredWithoutnoiseValues=MeasuredWithoutnoiseValues))

    if PlotResult=='Y':
        print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled before stopping')
        Difference,ReconImage = performReconOnce(SavePath,TrainingInfo,Resolution,SizeImage,ImageType,CodePath,ImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues)
        TD = Difference/(SizeImage[0]*SizeImage[1])
        img=loadTestImage(CodePath,ImageSet,ImNum,ImageExtension,SimulationRun)
        print('')
        print('')
        print('######################################')
        print('Total Distortion = ' + str(TD))
        
        from plotter import plotAfterSLADSSimulation
        plotAfterSLADSSimulation(Mask,ReconImage,img)
        pylab.show()
