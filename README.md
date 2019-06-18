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

