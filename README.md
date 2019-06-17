# Human Motion Dataset in the Wild (MAT)
Repository for the code used in the project Human Motion Dataset in the Wild (MAT). 

---
## Contents
* LSTM Model
* 2
## LSTM Model
This folder contains the model based on LSTM's that was trained for predicting the task performed in each sequences. The model is build in Pytorch and is thought to be trained on GPU. It also contains the feature extractor explained in the project report. The static features are obtained with a pretrained model that is provided by Pytorch and the dynamic features are obtained using Numpy and OpenCV. The wrists and ankles detector on image are a Pytorch implementation of Realtime Multi-Person Pose Estimation By Zhe Cao, Tomas Simon, Shih-En Wei and Yaser Sheikh. To make it work, a pretrained model is needed and it can be found in the repository of the pytorch implementation listed below. 

* Original repository: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
* Pytorch implementation repository: https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
