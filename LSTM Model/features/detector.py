# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:44:37 2019

@author: BIEL
"""

import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from features.network.rtpose_vgg import get_model
from features.network.post import decode_pose, NMS
from features.training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from features.network import im_transform
from features.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from collections import OrderedDict
import random

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

jNames={ 0:'nose',	1:'neck', 2:'right_shoulder' ,3:'right_elbow' ,4:'right_wrist',
5:'left_shoulder' ,6:'left_elbow'	    ,7:'left_wrist'  ,8:'right_hip',
9:'right_knee'	, 10:'right_ankle',	11:'left_hip' ,  12:'left_knee',
13:'left_ankle'	, 14:'right_eye'	,    15:'left_eye'  , 16:'right_ear',
17:'left_ear' }

class EgoPartDetector():
    def __init__(self, cuda=False):
        weight_name = 'features/pose_model.pth'
        self.iscuda=cuda
        if weight_name=='features/pose_model_scratch.pth':
            if self.iscuda:
                old_state_dict = torch.load(weight_name)
            else:
                old_state_dict = torch.load(weight_name,map_location='cpu')
            state_dict=OrderedDict()
            for k,v in old_state_dict.items():
                name = k[7:]
                state_dict[name]=v
        elif weight_name=='features/pose_model.pth':
            if self.iscuda:
                state_dict= torch.load(weight_name)
            else:
                state_dict= torch.load(weight_name,map_location='cpu')
        
        self.model = get_model(trunk='vgg19')
        
        if cuda:
#            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.float()
            self.model = self.model.cuda()
        else:
#            self.model = torch.nn.DataParallel(self.model).cpu()
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.float()
            self.model = self.model.cpu()
            
        self.backend=self.model.model0
        self.block1=self.model.model1_2
        
    def forward(self, image_root, flip= None):
        img=cv2.imread(image_root)
        shape_dst = np.min(img.shape[0:2])
        f=600/shape_dst
        img=cv2.resize(img,(0,0),fx=f,fy=f)
        
        if flip!=None:
            img=cv2.flip(img,flip)
        multiplier=[0.20444444444444446]
        preprocess='rtpose'
        
        heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
        max_scale = multiplier[-1]
        max_size = max_scale * img.shape[0]
        # padding
        max_cropped, _, _ = im_transform.crop_with_factor(
            img, max_size, factor=8, is_ceil=True)
        batch_images = np.zeros(
            (len(multiplier), 3, max_cropped.shape[0], max_cropped.shape[1]))
        
        for m in range(len(multiplier)):
            scale = multiplier[m]
            inp_size = scale * img.shape[0]
    
            # padding
            im_croped, im_scale, real_shape = im_transform.crop_with_factor(
                img, inp_size, factor=8, is_ceil=True)
    
            if preprocess == 'rtpose':
                im_data = rtpose_preprocess(im_croped)
    
            elif preprocess == 'vgg':
                im_data = vgg_preprocess(im_croped)
    
            elif preprocess == 'inception':
                im_data = inception_preprocess(im_croped)
    
            elif preprocess == 'ssd':
                im_data = ssd_preprocess(im_croped)
    
            batch_images[m, :, :im_data.shape[1], :im_data.shape[2]] = im_data
        
        if self.iscuda:
            batch_var = torch.from_numpy(batch_images).cuda().float()
        else:
            batch_var = torch.from_numpy(batch_images).cpu().float()
        output2 = self.block1(self.backend((batch_var)))
        heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
        
        for m in range(len(multiplier)):
            scale = multiplier[m]
            inginp_size = scale * img.shape[0]
        
            # padd
            im_cropped, im_scale, real_shape = im_transform.crop_with_factor(
                img, inp_size, factor=8, is_ceil=True)
            heatmap = heatmaps[m, :int(im_cropped.shape[0] /
                               8), :int(im_cropped.shape[1] / 8), :]
            heatmap = cv2.resize(heatmap, None, fx=8, fy=8,
                                 interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
            heatmap = cv2.resize(
                heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        return heatmap_avg
    def forward2(self,image_root):
        
        h1=self.forward(image_root)
        h2=self.forward(image_root,flip=0)
      
        
        h2=cv2.flip(h2,0)

        h=(h1+h2)/2
        
        return h
    
    def forward3(self,image_root):
        
        h1=self.forward(image_root)
        h2=self.forward(image_root,flip=0)
        h3=self.forward(image_root,flip=1)
        
        h2=cv2.flip(h2,0)
        h3=cv2.flip(h3,1)
        h=(h1+h2+h3)/3
        
        return h
        
def changeProbability(x,y,w,h,prob,jointTag):
    mult=1
    if jointTag=='right_wrist':
        mult=1.5 if x>=w/2 else 0.75 
    elif jointTag=='left_wrist':
        mult=1.5 if x<=w/2 else 0.75 
    elif jointTag=='right_ankle':
        mult=1.5 if x>=w/2 else 0.75 
    elif jointTag=='left_ankle':
        mult=1.5 if x<=w/2 else 0.75 
    return prob*mult
    
def findJoints(heatmaps,dimensions, verbose=False, detectClose=True):
    h,w=dimensions
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    joints=NMS(param,heatmaps)
    for i in range(len(joints)):
        joints[i]=[joints[i], jNames[i]]
    
    #delete the joints that were not found
    joints=list(filter(lambda x: len(x[0])>0, joints))
    #delete those unwanted joints
    wanted=['right_wrist', 'left_wrist','right_ankle', 'left_ankle']
    joints=list(filter(lambda x: x[1] in wanted, joints))
    
    #Adapt probabilities
    
    for i in range(len(joints)):
        x,y,prob,_ = joints[i][0][0]
        tag=joints[i][1]
        joints[i][0][0][2]=changeProbability(x,y,w,h,prob,tag)
    #detect if two segments are two close
    
    if detectClose:
        escape=False
        keep=True
        while keep:
            for i in range(len(joints)):
                xi,yi,probi,_ = joints[i][0][0]
                namei=joints[i][1]
                for j in range(i+1,len(joints)):
                    xj,yj,probj,_=joints[j][0][0]
                    namej=joints[j][1]
                    vector=np.array([xi,yi])-np.array([xj,yj])
                    distance=np.linalg.norm(vector)
                    if distance<0.1*min(h,w):
                        if probi<=probj:
                            del joints[i]
                        else:
                            del joints[j]
                        if verbose:
                            print('Close joints detected. [{},{}] and [{},{}]'.format(namei,probi,namej,probj))
                        escape=True
                        break
                if escape:
                    break
            if not escape:
                keep=False
            else:
                escape=False
                
    
                        
    return joints
def applyHeatmap(heatmaps,image_root,indices, save=True):
    try:
        indices=list(indices)
    except:
        indices=[indices]
    
    mxDim=500
    img=cv2.imread(image_root)
    h,w,_ = img.shape
    f=mxDim/max(h,w)
    img=cv2.resize(img,(0,0),fx=f,fy=f)
    
    if not os.path.isdir('Processed'):
        os.mkdir('Processed')
    
    heatmaps=heatmaps.transpose(2,0,1)
    for i in indices:
        
        hnp=heatmaps[i]
        hnp=255-hnp*255
        hnp=hnp.astype('uint8')
        
        
        colored=cv2.applyColorMap(hnp, cv2.COLORMAP_JET)
        hc,wc,_ = colored.shape
        colored=cv2.resize(colored,(0,0),fx=f*w/wc,fy=f*h/hc)
    #    colored=colored/255
        
    
        maps = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
        
        if save:
            fname=os.path.splitext(os.path.basename(image_root))[0]
            fname='Processed/{}_{}.jpg'.format(fname,i)
            cv2.imwrite(fname,255-maps*255)
        else:
            return maps

def get_msg_orientation(x,y,w,h):
    ori=[1,1]
    ori[0]=1 if x<=w//2 else 0
    ori[1]=1 if y<=h//2 else -1
    
    
    return ori

def getFeatures(heatmaps, image_root):
    img=cv2.imread(image_root)
    h,w,_ = img.shape
    hh,wh,_ = heatmaps.shape
    joints=findJoints(heatmaps,(hh,wh))
    
    tags=['right_wrist','left_wrist','right_ankle', 'left_ankle']
    order=[0,1,2,3]
    tags=[tags[i] for i in order]
    
    features=[]
    for tag in tags:
        pos=[0,0]
        for i in joints:
            if i[1]==tag:
                x,y,_,_ = i[0][0]
                x=x/wh
                y=y/hh
                pos=[x,y]
                break
        features+=pos
    features=np.array(features, dtype='float')
    return features
def applyJoint(heatmaps, image_root, colored=False, msg=False):
    if colored:
        img=applyHeatmap(heatmaps,image_root,[18], save=False)
    else:
        img=cv2.imread(image_root)
    
    h,w,_ = img.shape
    
    font_scale=max(h,w)*0.001
    font=cv2.FONT_HERSHEY_SIMPLEX
    text_color=(255,255,255)
    lineType=2
    hh,wh,_ = heatmaps.shape
    joints=findJoints(heatmaps,(hh,wh))
    
    if len(joints)!=0:
        
        for i in range(len(joints)):
            randcol=np.uint8(np.random.uniform(0,255,3))
            color=tuple(map(int,randcol))
            
            x,y,_,_ = joints[i][0][0]
            x=int(x*w/wh)
            y=int(y*h/hh)
            img=cv2.circle(img,(x,y),int(max(h,w)*0.01),color, thickness=-1)
            if msg:
                txt=joints[i][1]
                ori=get_msg_orientation(x,y,w,h)
                ori=np.array(ori)
                
                distance=min(h,w)*0.1*(0.5+0.5*random.random())
                text_position=np.array([x,y])+ori*(distance)
                pos=tuple(text_position.astype('uint16'))
        
            
                cv2.putText(img,txt,pos,font,font_scale,text_color,lineType)
                cv2.line(img,(x,y),pos,color)
        
    else:
        print('No body parts found in {}'.format(image_root))
    
    fname=os.path.splitext(os.path.basename(image_root))[0]
    fname='Processed/{}_joints.jpg'.format(fname)
    cv2.imwrite(fname,img)