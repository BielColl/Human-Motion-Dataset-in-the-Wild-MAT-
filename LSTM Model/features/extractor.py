# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:33:57 2019

@author: BIEL
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from features.staticExtractor import Img2Vec
from features.optical_flow import getStackOfImages, getSeqVector
from features.detector import EgoPartDetector, getFeatures
from PIL import Image
import os

def changeFrameNumber(root,number):
    base_root,name=os.path.split(root)
    frame,ext=os.path.splitext(name)
    frame=frame[:frame.find('_')+1]+str(number)
    return os.path.join(base_root,frame+ext)
def getNumber(root):
    base_root,name=os.path.split(root)
    frame,ext=os.path.splitext(name)
    return int(frame[frame.find('_')+1:])
def makeList(root, nHomographies):
    n=getNumber(root)
    roots=[]
    for i in range(n-nHomographies,n):
        roots.append(changeFrameNumber(root,i))
    roots.append(root)
    return roots
class FeatureExtractor():
    def __init__ (self, cuda=False, n_static=512, nHomographies=15):
        
        self.staticExt = Img2Vec(cuda=cuda,layer_output_size=n_static)
        self.det=EgoPartDetector(cuda=cuda)
        self.nHomographies=nHomographies
    def makeStack(self,root_list):
        stack=getStackOfImages(root_list)
        return stack
    def get_vec(self,frameRoot, show=False, verbose=True):
        
        root_list=makeList(frameRoot, self.nHomographies)
        if type(root_list[0])==tuple:
            root_list=[i[0] for i in root_list]
        # The list of frames containe n previous frames and the current frame
        
        if verbose: print('Doing '+frameRoot+'...')
        if verbose: print('Getting static features')
        #Get static features
        img = Image.open(frameRoot)
        staticVec = self.staticExt.get_vec(img)
        staticVec=torch.tensor(staticVec).float()
        
        if verbose: print('Getting dynamic features')
        #Geat dynamic features
        stack=self.makeStack(root_list)
        dynamicVec=getSeqVector(stack,show=show, qualityLevel=0.3)
        dynamicVec=dynamicVec[0].float()
        
        if verbose: print('Getting parts features')
        #Get hands, feet position
        
        h=self.det.forward(frameRoot)
        features=getFeatures(h,frameRoot)
        partsVec=torch.tensor(features).float()
        vec=torch.cat((staticVec,dynamicVec))
        vec=torch.cat((vec,partsVec))
        
        return vec