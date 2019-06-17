# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:17:29 2019

@author: BIEL
"""

from features.optical_flow import homography
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

def getStackOfImages(listofroots):
    trans=transforms.ToTensor()
    tensors=[]
    for root in listofroots:
        img=Image.open(root)
        imgt=trans(img)
        imgt=imgt.unsqueeze(0)
        tensors.append(imgt)
    return torch.cat(tuple(tensors))
    
def getSeq(frames,maxCorners=100,qualityLevel=0.3):
    # Frames is a sequence of n frames already imported as pytorch tensors
    # The goal of this functions is to obtain a sequency of n-1 homography 
    # matrices between the frames
    lH=[]
    for i in range(frames.shape[0]-1):
        img1=np.transpose(np.array(frames[i]),(1,2,0))*255
        img2=np.transpose(np.array(frames[i+1]),(1,2,0))*255
        
        img1=img1.astype('uint8')
        img2=img2.astype('uint8')

        
        H,p0,p1=homography(img1,img2,maxCorners=maxCorners, qualityLevel=qualityLevel)
        if H is None:
            H=np.eye(3)
        H=torch.tensor(H)
        H=H.unsqueeze(0)
        lH.append(H)

    return torch.cat(tuple(lH))

def getSeqVector(frames,maxCorners=100,qualityLevel=0.3):
    #Creates a 3D tensor of all the homographies (each slice in the first dimesnion
    # corresponds to a homography)
    #Then it unfold it in a one_dimensional vector in order: depth, left2right, top2bottom
    lH=getSeq(frames, maxCorners=maxCorners, qualityLevel=qualityLevel)
    v1,v2,v3=lH.shape
    vector=lH.view(-1,v1*v2).view(-1,v1*v2*v3)
    return vector