# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:29:48 2019

@author: BIEL
"""

import matplotlib.pyplot as plt
import numpy as np
import time

class predictionGraph():
    def __init__ (self):
        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.first=True
    def update(self, ypred, ytruth):
        print(len(ypred))
        print(len(ytruth))
        x=np.arange(len(ypred))
        if self.first:
            self.line1,= self.ax1.plot(x,ypred, 'r-', label='Predicted')
            self.line2,= self.ax1.plot(x,ytruth, 'b-',label='Truth')
            self.first=False
        else:
            self.line1.set_ydata(ypred)
            self.line2.set_ydata(ytruth)
        
        self.fig.canvas.draw()
        self.ax1.legend(loc='upper right')
        plt.pause(0.05)
class UpdatingGraph():
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
    def update(self,y):
        x=np.arange(len(y))
        
        self.line1,= self.ax1.plot(x,y, 'r-')
       
            
        
        self.fig.canvas.draw()
        plt.pause(0.05)