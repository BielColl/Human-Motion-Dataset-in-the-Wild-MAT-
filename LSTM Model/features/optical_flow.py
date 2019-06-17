# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:43:00 2019

@author: BIEL
"""

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

def opticalFlow(frame1, frame2, maxCorners,qualityLevel, show=False):
    #frames enter as numpy arrays
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = maxCorners,
                          qualityLevel = qualityLevel,
                          minDistance = 7,
                          blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0,255,(maxCorners,3))
    
    # Take first frame and find corners in it
    
    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1.copy()) 
    
    
    frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
#    p0 = good_new.reshape(-1,1,2)
    
    if show:
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame2 = cv2.circle(frame2,(a,b),5,color[i].tolist(),-1)
    
        img = cv2.add(frame2,mask)
        
        f=1
        cv2.imshow('test', cv2.resize(img,(0,0),fx=f, fy=f))
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.imwrite('test.jpg',img)
    
    return good_old, good_new

def opticalFlowMultiple(listofframes,maxCorners,qualityLevel, show=False):
        #frames enter as numpy arrays
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = maxCorners,
                          qualityLevel = qualityLevel,
                          minDistance = 7,
                          blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0,255,(maxCorners,3))
    
    # Take first frame and find corners in it
    frame1=listofframes[0]
    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1.copy()) 
    
    corresp=[] #list of the points throughout the video

    for i in range(1,len(listofframes)):
        
        frame2=listofframes[i]
        frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is None or p0 is None or len(p1)<4 or len(p0)<4:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        if len(good_old)==0:
                good_old2=np.vstack((good_old,np.array([0.,0.])))
                good_new2=np.vstack((good_new,np.array([0.,0.])))
                corresp.append((good_old2,good_new2))
        else:
            corresp.append((good_old,good_new))
        
        
        if show:
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        
        
    if show:
        frame2=frame2.copy()
        frame2 = cv2.circle(frame2,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame2,mask)
        f=1
        
        #fix channels
        red=img[:,:,2].copy()
        blue=img[:,:,0].copy()
        img[:,:,0]=red
        img[:,:,2]=blue
        
        cv2.imshow('test', cv2.resize(img,(0,0),fx=f, fy=f))
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.imwrite('test.jpg',img)
    
    return corresp

def homography(frame1, frame2, maxCorners=100,qualityLevel=0.2, show=False):
    
    p0,p1=opticalFlow(frame1,frame2, maxCorners=maxCorners,qualityLevel=qualityLevel, show=show)
    H,_=cv2.findHomography(p0,p1, method=cv2.RANSAC)
    if H is None:
        H=np.eye(3)
    return H,p0,p1

def getStackOfImages(listofroots):
    trans=transforms.ToTensor()
    tensors=[]
    for root in listofroots:
        img=Image.open(root)
        imgt=trans(img)
        imgt=imgt.unsqueeze(0)
        tensors.append(imgt)
    return torch.cat(tuple(tensors))
    
def getSeq(frames,maxCorners=100,qualityLevel=0.3, show=False):
    # Frames is a sequence of n frames already imported as pytorch tensors
    # The goal of this functions is to obtain a sequency of n-1 homography 
    # matrices between the frames
    
    lframes=[]
    for i in range(frames.shape[0]):
        img1=np.transpose(np.array(frames[i]),(1,2,0))*255
        img1=img1.astype('uint8')
        lframes.append(img1)
    
    correspondences=opticalFlowMultiple(lframes,maxCorners,qualityLevel, show=show)

    lH=[]  

    for i in range(len(correspondences)):
        p0,p1=correspondences[i]
        H,_=cv2.findHomography(p0,p1, method=cv2.RANSAC)
        if H is None: H=np.eye(3)
        H=torch.tensor(H)
        H=H.unsqueeze(0)
        lH.append(H)
        
    return torch.cat(tuple(lH))

def getSeqVector(frames,maxCorners=100,qualityLevel=0.3,show=False):
    #Creates a 3D tensor of all the homographies (each slice in the first dimesnion
    # corresponds to a homography)
    #Then it unfold it in a one_dimensional vector in order: depth, left2right, top2bottom
    lH=getSeq(frames, maxCorners=maxCorners, qualityLevel=qualityLevel,show=show)
    v1,v2,v3=lH.shape
    vector=lH.view(-1,v1*v2).view(-1,v1*v2*v3)
    return vector