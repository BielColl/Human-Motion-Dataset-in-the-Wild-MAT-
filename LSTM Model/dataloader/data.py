# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:14:24 2019

@author: BIEL
"""
import glob
import os
import csv
import torch
def getSeparation(seed):
    
    if seed=='overfit':
        tt=[1,2,3,5,6,7]
    elif seed=='allTrain':
        tt=[]
    elif seed <=0:
        tt=[1,5]
    elif seed ==1:
        tt=[2,6]
    elif seed==2:
        tt=[3,6]
    elif seed==3:
        tt=[4,6]
    elif seed==4:
        tt=[4,5]
    else:
        tt=[3,5]
        
    tr=list(filter(lambda x: x not in tt, range(1,8)))  
    test=['SUBJECT'+str(i) for i in tt]
    train=['SUBJECT'+str(i) for i in tr]
    d={'train':train,'test':test}
    return d
def getCSVNames(subdir, target='IND'):
    root,name=os.path.split(subdir)
    csvName=target+'_'+name+'.csv'
    return os.path.join(subdir,csvName)
def getSubject(subdir):
    name=os.path.basename(subdir)
    l=name.split('_')
    subject=l[1]
    return subject
def getAvailableDATA(datasetRoot, target='IND'):
    subdirs=glob.glob(datasetRoot+'/*')
    csvFiles=list(map(lambda x: getCSVNames(x, target=target), subdirs))
    if not all(os.path.isfile(i) for i in csvFiles):
        raise ValueError('Missing csv')
    csvFiles=list(map(lambda x: {'subject':getSubject(x), 'root': x}, csvFiles))
    return csvFiles

def separateData(datasetRoot, seed, target='IND'):
    csvFiles=getAvailableDATA(datasetRoot, target=target)
    separation=getSeparation(seed)
    test=[]
    train=[]
    for file in csvFiles:
        if file['subject'] in separation['train']:
            train.append(file)
        else:
            test.append(file)
    return train,test

def loadAllData(csvFile, low_cut=None):
    frames=[]
    poses=[]
    root,_=os.path.split(csvFile)
    with open(csvFile, 'r') as f:
        r=csv.reader(f,delimiter=',')
        for row in r:
            frame=os.path.join(root,row[0])
            pose=int(row[1])
            frames.append(frame)
            poses.append(pose)
    if type(low_cut)==int:
        frames=frames[low_cut:]
        poses=poses[low_cut:]
    elif type(low_cut)==list and len(low_cut)>=2:
        frames=frames[low_cut[0]:low_cut[1]]
        poses=poses[low_cut[0]:low_cut[1]]
    return frames,poses

def loadAllData_FEAT(csvFile, low_cut=None):
    feats=[]
    targets=[]
    root,_=os.path.split(csvFile)
    with open(csvFile, 'r') as f:
        r=csv.reader(f,delimiter=',')
        for row in r:
            feat=row[:-1]
            tar=row[-1]
            feats.append(feat)
            targets.append(tar)
    if type(low_cut)==int and len(targets)>low_cut:
        feats=feats[:low_cut]
        targets=targets[:low_cut]
    elif type(low_cut)==list and len(low_cut)>=2 and len(targets)>low_cut[1]:
        feats=feats[low_cut[0]:low_cut[1]]
        targets=targets[low_cut[0]:low_cut[1]]
    return feats,targets
def getBatches(frames,poses,batch_size, nHomographies):
    groupedFrames=[]
    for i in range(nHomographies+1, len(frames)):
        roots=frames[i-nHomographies-1:i]
        groupedFrames.append(roots)
    posesUn=poses[nHomographies+1:]
    posesBatch=[]
    framesBatch=[]
    for i in range(0,len(posesUn),batch_size):
        if len(poses)>i+batch_size:
            posesBatch.append(torch.cat(posesUn[i:i+batch_size]))
            framesBatch.append(groupedFrames[i:i+batch_size])
        else:
            posesBatch.append(torch.cat(posesUn[i:]))
            framesBatch.append(groupedFrames[i:])
    return framesBatch, posesBatch
            
