# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:35:00 2019

@author: BIEL
"""

import torch
from torch import nn
from features.extractor import FeatureExtractor
from torch.autograd import Variable
from torch.nn import functional as F
class LSTM (nn.Module):
    def __init__ (self, n_static, n_dynamic, hidden_dim, output_dim,batch_size, 
                  num_layers=2):
        self.embed_size=256
        self.n_static=n_static
        self.n_dynamic=n_dynamic
        #Now n_dynamic is per default 15
        super(LSTM, self).__init__()
        self.input_dim = n_static+n_dynamic*9+8
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim+self.embed_size, self.hidden_dim, self.num_layers, dropout=1)
    
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
        #Mapping the embed
        self.map = nn.Linear(output_dim, self.embed_size)
        self.embed=torch.zeros(1,256)
        #Create the feature extractor
        self.featureExt=FeatureExtractor(cuda=False,n_static=n_static, nHomographies=self.n_dynamic)
     
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    def prepareImages(self, inp, verbose):
        first=True
        for listOfRoots in inp:
            vec=self.featureExt.get_vec(listOfRoots, verbose=verbose)
            if first:
                vectors=vec.unsqueeze(0)
                first=False
            else:
                vectors=torch.cat((vectors,vec.unsqueeze(0)))
        return vectors
    def forward(self, inps, verbose=False):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        if verbose:
            print('Extracting features...')
        inputVecs=self.prepareImages(inps, verbose)
        first=True
        
        for i in range(inputVecs.shape[0]):
            inputVec=torch.cat((inputVecs[i,:].unsqueeze(0), self.embed), dim=1)
            lstm_out, self.hidden = self.lstm(inputVec.unsqueeze(0))
            y_pred = self.linear(lstm_out)[0]
            self.embed=self.map(F.log_softmax(y_pred, dim=1))
            if first:
                outputs=y_pred
                first=False
            else:
                outputs=torch.cat((outputs,y_pred))
            
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        
#        y_pred = F.log_softmax(y_pred, dim=1)
        return outputs
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
class LSTM_FEAT (nn.Module):
    def __init__ (self, n_static, n_dynamic, hidden_dim, output_dim,batch_size,embed_size,
                  num_layers=2, cuda=False):
        
        if cuda:
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")
        self.cuda=cuda
        self.embed_size=embed_size
        self.n_static=n_static
        self.n_dynamic=n_dynamic
        #Now n_dynamic is per default 15
        super(LSTM_FEAT, self).__init__()
        self.input_dim = n_static+n_dynamic*9+8
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim+self.embed_size, self.hidden_dim, self.num_layers)
    
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
        #Mapping the embed
        self.map = nn.Linear(output_dim, self.embed_size)
        self.embed=torch.zeros(1,self.embed_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
    def detach_hidden(self):
        
        self.hidden=(self.hidden[0].detach(), self.hidden[1].detach())
        
#        self.hidden=Variable(self.hidden.data, requires_grad=True)
    def reset_grad(self):
        self.lstm.zero_grad()
        self.linear.zero_grad()
        self.map.zero_grad()
    def forward(self, inputVecs,device, verbose=False):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        if verbose:
            print('Extracting features...')
        first=True
        
        for i in range(inputVecs.shape[0]):
            inputVec=torch.cat((inputVecs[i,:].unsqueeze(0), self.embed.to(device)), dim=1).to(device)
            lstm_out, self.hidden = self.lstm(inputVec.unsqueeze(0))
            y_pred = self.linear(lstm_out)[0]
            self.embed=self.map(F.log_softmax(y_pred, dim=1))
            
            if first:
                outputs=y_pred
                first=False
            else:
                outputs=torch.cat((outputs,y_pred))
            
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        
#        y_pred = F.log_softmax(y_pred, dim=1)
        self.hidden = repackage_hidden(self.hidden)
        return outputs

def prepareFeats(inp):
    first=True
    for vec in inp:
        if first:
            vectors=vec.unsqueeze(0)
            first=False
        else:
            vectors=torch.cat((vectors,vec.unsqueeze(0)))
    return vectors     
            
