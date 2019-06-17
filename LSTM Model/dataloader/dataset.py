# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:02:44 2019

@author: BIEL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
from torchvision import transforms
from dataloader.data import separateData, loadAllData,loadAllData_FEAT
import os


class DatasetEGOPOSE(Dataset):
    def __init__(self, mainRoot, tag, seed=0, nHomographies=15):
        self.nHomographies=nHomographies
        train,test = separateData(mainRoot, seed)
        if tag=='test':
            self.listCSV=test
        elif tag=='train':
            self.listCSV=train
        else:
            raise ValueError ('Tag should be train or test')
        self.sequences=[loadAllData(i['root'], low_cut=nHomographies) for i in self.listCSV]
        self.lengths=[len(x[0]) for x in self.sequences]
    def __len__(self):
        return len(self.listCSV)
    def __getitem__(self,indx):
        #Return sequence
        elem=None
        if type(indx)==torch.Tensor: indx=int(indx)
        for i in range(len(self.sequences)):
            if indx<self.lengths[i]:
                elem=(self.sequences[i][0][indx],self.sequences[i][1][indx])
                break
            else:
                indx-=self.lengths[i]
        if elem==None: 
            raise ValueError ('Index out of range')
        else:
            frame,pose=elem
        return frame,int(pose)

class DatasetEGOPOSE_FEAT(Dataset):
    def __init__(self, mainRoot, tag, seed=0, nHomographies=15, target='IND', limiter=None):
        self.nHomographies=nHomographies
        train,test = separateData(mainRoot, seed, target)
        self.target=target
        if tag=='test':
            self.listCSV=test
        elif tag=='train':
            self.listCSV=train
        else:
            raise ValueError ('Tag should be train or test')
        if limiter!=None:
            self.listCSV=self.listCSV[:limiter]
        if target=='CLASS':
            low_cut=[1000,50000]
        else:
            low_cut=None
        self.sequences=[loadAllData_FEAT(i['root'], low_cut=low_cut) for i in self.listCSV]
        self.lengths=[len(x[0]) for x in self.sequences]
    def __len__(self):
        return len(self.listCSV)
    def __getitem__(self,indx):
        #Return sequence
        elem=None
        if type(indx)==torch.Tensor: indx=int(indx)
        for i in range(len(self.sequences)):
            if indx<self.lengths[i]:
                elem=(self.sequences[i][0][indx],self.sequences[i][1][indx])
                break
            else:
                indx-=self.lengths[i]
        if elem==None: 
            raise ValueError ('Index out of range')
        else:
            frame,pose=elem
            
        return torch.tensor([float(i) for i in frame]),int(pose)

class CustomSampler(Sampler):
    def __init__ (self, dataset, batch_size=5, drop_last=False, shuffle=False):
        self.lengths=[0]+dataset.lengths
        for i in range(1,len(self.lengths)): self.lengths[i]+=self.lengths[i-1]
        self.sequences=[list(range(self.lengths[i-1],self.lengths[i])) for i in range(1,len(self.lengths))]
        self.batch_size=batch_size
        self.batches=[y[x:x+self.batch_size] for y in self.sequences  for x in range(0, len(y), self.batch_size)]
        self.drop_last=drop_last
        if self.drop_last:
            self.batches=list(filter(lambda x: len(x)==batch_size, self.batches))
        self.batches=torch.tensor(self.batches)
        if shuffle:
            self.batches=self.batches[torch.randperm(self.batches.size()[0])]
    def __iter__ (self):
        return iter(self.batches)
    def __len__ (self):
        return len(self.batches)
    
class CustomSampler_FEAT(Sampler):
    def __init__ (self, dataset, batch_size=5, drop_last=False, shuffle=False):
        self.lengths=[0]+dataset.lengths
        for i in range(1,len(self.lengths)): self.lengths[i]+=self.lengths[i-1]
        self.sequences=[list(range(self.lengths[i-1],self.lengths[i])) for i in range(1,len(self.lengths))]
        self.batch_size=batch_size
        self.batches=[y[x:x+self.batch_size] for y in self.sequences  for x in range(0, len(y), self.batch_size)]
        self.drop_last=drop_last
        if self.drop_last:
            self.batches=list(filter(lambda x: len(x)==batch_size, self.batches))
        self.batches=torch.tensor(self.batches)
        if shuffle:
            self.batches=self.batches[torch.randperm(self.batches.size()[0])]
    def __iter__ (self):
        return iter(self.batches)
    def __len__ (self):
        return len(self.batches)

