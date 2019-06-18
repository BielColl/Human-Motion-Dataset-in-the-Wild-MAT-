# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:57:29 2019

@author: BIEL
"""

import numpy as np
import cv2
import glob
import random

def getParametersFromImages(list_of_images,chess_dimensions, square_size,f=0.5,saveName=None, controled=True):
    #list_of_images contain all the images that will be used for calibrating
    #chess dimensions contains the number of corners in height and in width [IN THIS ORDER]
    #square_size contains the lenght of the side of the square in whatever unit you want (the matrix will be in that units)
    #f is the resizing factor for the images that will be shown during the calibration
    #if saveName is a string, the parameters will be saved as an npz file
    
    nch=chess_dimensions[0] #number corners in height
    ncw=chess_dimensions[1] #number corners in width
    
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    
    #prepare object points, like (0,0,0), (1,0,0)... (ncw,5,0) --> it's a nchxncw corner grid
    objp=np.zeros((ncw*nch,3), np.float32)
    objp[:,:2]=np.mgrid[0:nch,0:ncw].T.reshape(-1,2)
    objp=objp*square_size #cada quadrat fa 40 mm
    
    #Arrays to store object points and image points from all images
    objpoints=[] #3d points inr eal world space
    imgpoints=[] #2d points in image plane
    i=1
    
    direct=True
    if type(list_of_images[0])==str:
        direct=False
    c=0
    for img in list_of_images:
        print('Finding corners in file {}'.format(i))
        if not direct:
            img=cv2.imread(img)
        
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        #find the chess board corners
        ret,corners = cv2.findChessboardCorners(gray, (nch,ncw), None)
        
        #if found, add object points, image points (after refining them)
        
        if ret==True:
            corners2= cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            img=cv2.drawChessboardCorners(img,(nch,ncw),corners2,ret)
            cv2.imwrite('saved/saved_{}.jpg'.format(c), img)
            c+=1
            accepted=True
            while(1):
                cv2.imshow('img',cv2.resize(img,(0,0),fx=f,fy=f))
                k=cv2.waitKey(500)
                if not controled: break
                if k==ord('q'):
                    accepted=False

                    break
                elif k==-1:
                    continue
                else:
                    break
                
            if accepted:
                objpoints.append(objp)
                
                imgpoints.append(corners2)
                        
        else: print('Error finding corners of file {}'.format(i))
        i+=1
    
    
    cv2.destroyAllWindows()
    
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,
                                               gray.shape[::-1],None,None)
    
    if type(saveName)==str:
        np.savez(saveName, ret,mtx, dist,rvecs,tvecs)
    
    return mtx,dist

def cameraCalibration_Video(video_root,chess_dimensions, square_size,f=0.5,saveName=None,
                            time_cuts=None, nframes=50, controled=True):
    # New input variables:
    # - Video root is the root of the video (in mp4)
    # - If time_cuts can tell the periods of time in which the frames are extracted
    #   It has to be a list of tuples, each tuple indicating the period of time
    # - n_frames is the number of frames used for calibration
    
    print('Extracting frames from {}'.format(video_root))
    capvid=cv2.VideoCapture(video_root)
    totalframes=capvid.get(cv2.CAP_PROP_FRAME_COUNT)
    fps=capvid.get(cv2.CAP_PROP_FPS)
    
    if type(time_cuts)==list and type(time_cuts[0])==tuple:
        frames_range=[]
        for r in time_cuts:
            a=tuple(map(lambda x: int(x*fps),r))
            frames_range.append(a)
    else:
        frames_range=[(0,int(totalframes))]
    
    avail=list(map(lambda x: list(range(x[0],x[1]+1)),frames_range))
    avail=[j for i in avail for j in i]
    
    if len(avail)<nframes:
        print('CAUTION: only {} frames available instead of {}'.format(len(avail), nframes))
        nframes=len(avail)
        
    frames_to_get=[]
    for i in range(nframes):
        frame=avail.pop(random.randint(0,len(avail)-1))
        frames_to_get.append(frame)
    
    images=[]
    i=0
    for frame in frames_to_get:
        capvid.set(1,frame)
        ret,frame=capvid.read()
        if ret:
            images.append(frame)
        else:
            break
        
        if i%10==0:
            print('Extracted {} frames out of {}'.format(i+1,nframes))
        i+=1
    
    print('Frames extracted')
    print('Obtaining intrinsic parameters')
    mtx,dist=getParametersFromImages(images,chess_dimensions,square_size,f=f,controled=controled,saveName=saveName)
    
    return mtx,dist

def cameraCalibration_Images(images_root,chess_dimensions, square_size,f=0.25,saveName=None):
    # New variables:
    # - Images_root is a directory will all the frames to be used for calibration (jpg)
    
    images= glob.glob(images_root+'\*.jpg')
    mtx,dist=getParametersFromImages(images,chess_dimensions,square_size,f=f,controled=controled,saveName=saveName)
    
    return mtx,dist
    