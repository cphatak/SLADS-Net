#! /usr/bin/env python3

class TrainingInfo:
    def initialize(self,ReconMethod,FeatReconMethod,p,NumNbrs,FilterType,FilterC,FeatDistCutoff,MaxWindowForTraining,*args):
        self.ReconMethod = ReconMethod
        self.FeatReconMethod = FeatReconMethod
        self.p = p
        self.NumNbrs = NumNbrs        
        self.FilterType = FilterType
        self.FilterC = FilterC
        self.FeatDistCutoff = FeatDistCutoff
        self.MaxWindowForTraining=MaxWindowForTraining
        if args:
            self.PAP_Iter=args[0]
            self.PAP_Beta=args[1]
            self.PAP_InitType=args[2]
            self.PAP_ScaleMax=args[3]
class InitialMask:
    def initialize(self,RowSz,ColSz,MaskType,MaskNumber,Percentage):
        self.RowSz = RowSz
        self.ColSz = ColSz
        self.MaskType = MaskType
        self.MaskNumber = MaskNumber
        self.Percentage = Percentage

class StopCondParams:
    def initialize(self,Beta,Threshold,JforGradient,MinPercentage,MaxPercentage):
        self.Beta = Beta
        self.Threshold = Threshold
        self.JforGradient = JforGradient
        self.MinPercentage = MinPercentage
        self.MaxPercentage = MaxPercentage

class UpdateERDParams:
    def initialize(self,Do,MinRadius,MaxRadius,IncreaseRadiusBy):
        self.Do = Do
        self.MinRadius = MinRadius
        self.MaxRadius = MaxRadius
        self.IncreaseRadiusBy = IncreaseRadiusBy


class BatchSamplingParams:
    def initialize(self,Do,NumSamplesPerIter):
        self.Do = Do
        self.NumSamplesPerIter = NumSamplesPerIter

class EDSData:
    def initialize(self,NumSpectra,Folder,NoiseType,Noiselambda,ErrorSpectrumProb):
        self.NumSpectra = NumSpectra
        self.Folder = Folder
        self.NoiseType = NoiseType
        self.Noiselambda = Noiselambda
        self.ErrorSpectrumProb = ErrorSpectrumProb
    
class tfclfstruct:
    def __init__(self, sess, W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv):
        self.sess = sess
        self.W_conv1 = W_conv1
        self.W_conv2 = W_conv2
        self.W_fc1 = W_fc1
        self.W_fc2 = W_fc2
        self.W_fc3 = W_fc3
        self.W_fco = W_fco
        self.x_image = x_image
        self.x = x
        self.y_ = y_
        self.h_conv1 = h_conv1
        self.h_conv2 = h_conv2
        self.h_pool1 = h_pool1
        self.h_pool2 = h_pool2
        self.size_hp = size_hp
        self.h_flat = h_flat
        self.h_fc1 = h_fc1
        self.h_fc2 = h_fc2
        self.h_fc3 = h_fc3
        self.keep_prob = keep_prob
        self.h_fc1_drop = h_fc1_drop
        self.y_conv = y_conv
        
        
        
        
        
        
        
        
        
        