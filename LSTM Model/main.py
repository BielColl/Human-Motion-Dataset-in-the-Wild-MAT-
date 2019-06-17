# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:36:25 2019

@author: BIEL
"""
from network.models import LSTM, LSTM_FEAT,LSTM_FEAT_2,LSTM_FEAT_3, prepareFeats
import glob
import os
from dataloader import data, dataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from graphs import classes
import csv
from torch.autograd import Variable
import numpy as np

#Hyperparameters
target='IND'
datasetRoot='DATASET_FEAT'

staticFeatures=512
nHomographies=15

if target=='CLASS':
    outputDim=5
    hiddenDimension=128
    embed_size=8
    tagsFile='classTags.csv'
else:
    outputDim=500
    hidden2=512
    hiddenDimension=512
    embed_size=256
    tagsFile='indTags.csv'

batch_size=64
n_layers=2
LR=0.0001
n_epochs=20
seed='overfit'
dropout=0
cuda=True
graph=False
reset=True

savedModel='statedict/net_epoch_0.pth'
current_epoch=1

#Load loss weights
l=[]
if tagsFile=='classTags.csv':
    with open('classTags.csv','r') as f:
        r=csv.reader(f,delimiter=',')
        for row in r:
            l.append(row)
    total=sum([int(i[2]) for i in l])
    lossWeights=list(map(lambda x: (total-int(x[2]))/total, l))
    lossWeights=torch.tensor(lossWeights)
elif tagsFile=='indTags.csv':
    with open('indTags.csv','r') as f:
        r=csv.reader(f,delimiter=',')
        for row in r:
            l.append(row)
    total=sum([int(i[1]) for i in l])
    lossWeights=list(map(lambda x: (total-int(x[1]))/total, l))
#    lossWeights=total/(outputDim*np.array([int(x[1]) for x in l]))
    lossWeights=torch.tensor(lossWeights).float()
else:
    lossWeights=None
#Load Data
print('Loading Dataset')
train_dataset=dataset.DatasetEGOPOSE_FEAT(datasetRoot,'train', seed=seed, target=target)
samplerTrain=dataset.CustomSampler(train_dataset,batch_size, drop_last=True, shuffle=True)
train_loader=DataLoader(train_dataset,batch_sampler=samplerTrain)
train_iter=iter(train_loader)

#test_dataset=dataset.DatasetEGOPOSE_FEAT(datasetRoot,'test', seed=seed, target=target)
#samplerTest=dataset.CustomSampler(test_dataset,batch_size, drop_last=True, shuffle=True)
#test_loader=DataLoader(test_dataset,batch_sampler=samplerTest)
#test_iter=iter(test_loader)

print('Creating net')
net=LSTM_FEAT(staticFeatures, nHomographies,hiddenDimension, outputDim,batch_size,embed_size, 
                  num_layers=2, cuda=cuda, dropout=dropout)

if not reset and os.path.isfile(savedModel):
    net.load_state_dict(torch.load(savedModel))
if cuda:
    device=torch.device("cuda")
    lossWeights=lossWeights.to(device)
else:
    device=torch.device("cpu")


    
net=net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
if type(lossWeights)==torch.Tensor:
    loss_func = nn.CrossEntropyLoss(weight=lossWeights)  
else:
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
ll_loss_train=[]
ll_acc_train=[]
ll_loss_test=[]
ll_acc_test=[]
if graph:
    loss_graph=classes.UpdatingGraph()
first=True
for epoch in range(n_epochs):
    count=0
    total=int(0.85*len(train_iter))
    trc=1
    sl=0
    cc=0
    running_corrects=0
    if not any(True for i in train_iter):
            train_iter=iter(train_loader)
            print('Reseting train_iter')
    for frames,poses in train_iter:

        
#        net.reset_grad()
        print('\r [{}/{}] Training...'.format(trc,total), end='')
        inputVec=prepareFeats(frames)
#        net.hidden=net.init_hidden()
        if cuda:
            inputVec=inputVec.to(device)
            poses=poses.to(device)
  
        output=net(inputVec, device, verbose=False)                             # rnn output
        loss = loss_func(output, poses) 
        
        optimizer.zero_grad()
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        sl+=loss.detach().item()
        
        pred_y = torch.argmax(output, 1).data.cpu().numpy()
        target=poses.cpu().numpy()
        running_corrects += float((pred_y == target).astype(int).sum()) / float(target.size)
        trc+=1
        cc+=1
        if cc>=total:
            break

    print('')
    print(' Loss: {:.8f}  '.format(sl/cc))
    print(' Accuracy: {:.8f}  '.format(running_corrects/cc))
    print('')
    ll_loss_train.append(sl/cc)
    ll_acc_train.append(running_corrects/cc)
    if graph:
        loss_graph.update(ll_loss_train)

    print('')
    torch.save(net.state_dict(), 'statedict/ind_net_epoch_{}.pth'.format(epoch+current_epoch))
    with open('train.csv','w',newline='') as f:
        w=csv.writer(f,delimiter=',')
        w.writerow(ll_loss_train)
        w.writerow(ll_acc_train)
    if count % 1 == 0:
        total=int(0.15*len(train_iter))
        
#        if not any(True for i in test_iter):
#            test_iter=iter(test_loader)
#            print('Reseting test_iter')
        tcount=0
        accuracy=0
        lc=0
        with torch.no_grad():
            for tframes,tposes in train_iter:
                print('\r[{}/{}] Testing...'.format(tcount,total), end='')
                inputVec=prepareFeats(tframes)
                
                if cuda:
                    inputVec=inputVec.to(device)
                    tposes=tposes.to(device)
                
                toutput=net(inputVec, device,verbose=False)                 # (samples, time_step, input_size)
                pred_y = torch.argmax(output, 1).data.cpu().numpy()
                target=tposes.cpu().numpy()
                accuracy += float((pred_y == target).astype(int).sum()) / float(target.size)
                loss = loss_func(toutput, tposes)
                lc += loss.item()
                tcount+=1
        
        ll_loss_test.append(lc/tcount)
        ll_acc_test.append(accuracy/tcount)
        accuracy=accuracy/tcount
        
        with open('test.csv','w',newline='') as f:
            w=csv.writer(f,delimiter=',')
            w.writerow(ll_loss_test)
            w.writerow(ll_acc_test)
        print('')
        lc=lc/tcount
        print('Epoch: ', epoch, '| train loss: %.6f' % lc, '| test accuracy: %.6f ' % accuracy)
        ll_acc_test.append(accuracy)
        
    
    if epoch>2 and epoch == n_epochs//2:
        try:
            for g in optimizer.param_groups:
                g['lr']=0.0001
            print('Reduced LR')
        except:
            aaa=0
