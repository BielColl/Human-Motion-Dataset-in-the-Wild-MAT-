% Pose Projection Interactive App (Multiple frame PnP)
% Gabriel Coll Ribes
%-------------------------

%Checking if already uploaded MVNX file, otherwise, upload it
if exist('mvnx','var')~=1
    clearvars
    mvnx=load_mvnx('Helena_40.mvnx'); 
    %if one wants to use another file, delete the previous 
    %mvnx variable from the workspace
else
    clearvars -except mvnx
end

close all;

%Adding the needed paths
addpath(genpath('./PnPs'));
addpath('./Frames');

%Parameters
l=load('videocamera_calibration.mat'); mtx=l.mtx; dist=l.dist; clear l;
original_mtx=mtx;

sync=67;
vframes=[951,627];
dframes=vframes-sync+3;
test_frame=[160:2:250, 650:2:900];
skip=2; fps=60/skip;
gif_frames=160:skip:1020;
stamp=datestr(clock);
stamp=strrep(stamp,'-','_');
stamp=strrep(stamp,':','_');
stamp=strrep(stamp,' ','_');
gif_name=strcat("projection_",stamp);
max_iter=300; noise=0.03;

%Select initial imagepoints
avail_segments=["right wrist", "right elbow", "left wrist", "left elbow",...
                "right toe", "right ankle", "right knee", "right hip",...
                "left toe", "left ankle", "left knee", "left hip", "hip"];


%create bank of frames
bank=containers.Map('KeyType','double', 'ValueType', 'any');
for frame=vframes
    frameMap=containers.Map;
    imagename=strcat('Frames To Start/frame_',int2str(frame),'.jpg');
    image=imread(imagename);
    image=flip(image,1);
    image=flip(image,2);
    frameMap('image')=image;
    frameMap('vframe')=frame;
    frameMap('dframe')=frame-sync+3;
    s=size(image); height=s(1);width=s(2); clear s;
    
    selected=1:length(avail_segments);
    if frame==290
        selected=[1,2,5,9,10,11];
    elseif frame==124
        selected=[1,2,3,4,5,6,9,8,12,13];
    elseif frame==627
        selected=[1,2,3,4,5,7,8,9,12,13];
    elseif frame==647
        selected=[1,2,4,5,9,6,7,8,12,13];
    elseif frame==951
        selected=[1,2,3,5,9,7,11,8];
        
    elseif frame==759 || frame==740
        selected=[1,3,4,5,9,7,11];
    elseif frame==702
        selected=[1,3,4,5,9,7,11,8,12,13];
        
    end
    
    frameMap('selected')=selected;
    frameMap('tags')=avail_segments(selected);
    bank(frame)=frameMap;
end


%select image_points
f1=figure();
img_points=[];
obj_points=[];
ref_segment=24;
for i=vframes
    fbank=bank(i);
    a=selectImagePoints(fbank);
    img_points=[img_points;a];
    a=obtainObjPoints(mvnx,fbank,ref_segment);
    obj_points=[obj_points;a];
    hold on;
    pause(0.2);
    clf(f1);
end

close(f1);

%calculate R and T

[R,T,error]=iterativePnP(img_points,obj_points,mtx,0.002*min(height,width));

%Project results

f2=knownProjection(test_frame,sync,R,T,mvnx,ref_segment,mtx,1);
close(f2);
