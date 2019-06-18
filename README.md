# Human Motion Dataset in the Wild (MAT)
Repository for the code used in the project Human Motion Dataset in the Wild (MAT). 

## Contents
* LSTM Model
* Camera Calibration

## LSTM Model
This folder contains the model based on LSTM's that was trained for predicting the task performed in each sequences. The model is build in Pytorch and is thought to be trained on GPU. It also contains the feature extractor explained in the project report. The static features are obtained with a pretrained model that is provided by Pytorch and the dynamic features are obtained using Numpy and OpenCV. The wrists and ankles detector on image are a Pytorch implementation of Realtime Multi-Person Pose Estimation By Zhe Cao, Tomas Simon, Shih-En Wei and Yaser Sheikh. To make it work, a pretrained model is needed and it can be found in the repository of the pytorch implementation listed below. 

* [Original repository of the joint detector](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
* [Pytorch implementation repository of the joint detector](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)

Folder contents:

* *./features*: Subfolder with all the code related to the feature extractor
* *./graphs*: Subfolder containing the code for the matplotlib graphs for plotting loss and accuracy data during training
* *./network*: Subfolder with the different LSTM models: the one actually trained for the report, a version of th proposed network for pose estimation, and some variations
* *./dataloader*: Subfolder containing the dataloader needed to import the dataset for training and testing
* *./main.py*: Script with the training loop

## Camera calibration

This folder contains the Python script for the camera calibration, that was build using the OpenCV library for Python.

Folder contents:

* *./cameraCalibration.py*: Small library with several functions to obtain the intrinsic parameters of a camera. While a camera can be calibrated using a single image, if several images are used the results improve. As an input, one can use a set of individual pictures or a video. If the input is a video, the script itself will extract a set of frames to execute the calibration.
* *./cameraParameters.npz*: Example of the saved output of the calibration. It contains the intrinsic matrix and the distortion parameters.

## Projection

This folder contains the MATLAB scripts used for projecting MVN Awinda data onto the video frames. 

Folder main contents:

* *./poseProjectionMultiplePnP.m*: Main interactive app to project MVNX data to a sequence of frames. It starts by manually selecting the joints seen at the frame. To increase the number of points in the EPnP algorithm, several frames of the same sequence can be used. Then, the EPnP algorithm is run and the projections is shown on screen. If the results can be improved, the app lets the user redefine the manually selected points or manually tweak the camera pose while visualizing the changes in the projection. 

* *./iterativePnP.m*: Script that applies an EPnP algorithm to obtain the camera pose in the sequence. Because of the noise that could be introduced in the manually selected points by the user, this script applies several changes to the initial points. Minor translations are applied to each point at every iteration and, then, the EPnP algorithm is applied. Then, the points are reprojected using the R and T obtained and are compared with the points selected by the user, computing a projection error. The script saves the best set of initial points and returns the corresponding R and T matrices.

* *./addNoise2Points.m*: Script that adds noise to the initial points selected by the user, to apply the minor translations mentioned above.
* *./changeCoordinateSystem.m* Given a point expressed in the coordinate system A, the axis of another coordinate system B in A and the position of B in A, return the point expressed in B.  
* *./createProjectionsMP4.m*, *./createProjectionsGif.m*: Exports a projected sequence in MP4 or GIF, respectively.
* *./knownProjection.m*: Given a set of intrinsic and extrinsic parameters, shows on screen the resulting projected sequence. 
* *./undistort.m*: Undistorts a frame using the distortion coefficients of the camera.
* *./videocamera_calibration.mat*: Contains the intrinsic parameters of the camera used




