%Load MVNX file, if not done
if exist('mvnx','var')~=1
    clearvars
    mvnx=load_mvnx('Test_5_Updated.mvnx');
else
    clearvars -except mvnx
end

close all;
%Needed paths
addpath(genpath('./PnPs'));

%Parameters needed
width=1920; height=1080;
sync=74;
vframe=800;
frame=vframe-sync+3;
ref_segment=7;
off=0.5*width;  
noise=0; %noise as percent of min(width, height)

%Generate image
whiteImage=255*ones(height,width,3,'uint8');
f1=figure();
imshow(whiteImage);
hold on;
%Camera parameters
l=load('videocamera_calibration.mat'); mtx=l.mtx; dist=l.dist; clear l;

%Import 3DPose Data
pos_data = mvnx.subject.frames.frame(frame).position;
pos_data=reshape(pos_data,3,length(pos_data)/3)';
pos_data=pos_data*1000;

head_pos=pos_data(ref_segment,:);
quats=mvnx.subject.frames.frame(frame).orientation;
quats=reshape(quats,4,length(quats)/4)';
head_quat=quats(ref_segment,:);
index=1:length(pos_data(1:end-1,1));
obj_points=[];
for i =index
   row=pos_data(i,:);
   p=changeCoordinateSystem(row,head_quat,head_pos);
   obj_points=[obj_points;reshape(p,1,3)];
end

%Define camera
choose=1;

if choose==1   %at chest height, 45 degree view
    segment4height=5;
    cameraPos=pos_data(segment4height,:); 
    bodyAxes=quat2mat((quats(segment4height,:)));
    angle=(10*pi/180); radius=2000; extraElevation=0;
    cameraPos=cameraPos+cos(angle)*radius*bodyAxes(:,1)'+sin(angle)*radius*bodyAxes(:,2)';
    cameraPos=cameraPos+extraElevation*bodyAxes(:,3)';
    
    cameraAxes=rotationAxis(bodyAxes(:,3),pi)*rotationAxis(bodyAxes(:,3),angle)*bodyAxes;
    cameraAxes=rotationAxis(bodyAxes(:,3),-pi/2)*cameraAxes;
    cameraAxes=rotationAxis(cameraAxes(:,1),-pi/2)*cameraAxes;
    
    %All the previous was defined in the world coordinates
elseif choose==2 %simulated egocentric camera
    cameraPos=pos_data(7,:);
    headAxes=quat2mat((quats(7,:)));
    headLength=180; cameraAngle=(85*pi/180);
    elevationFromHead=200;
    cameraPos=cameraPos+headLength*headAxes(:,1)'+elevationFromHead*headAxes(:,3)';
    
    cameraAxes=rotationAxis(headAxes(:,3),-pi/2)*headAxes;
    cameraAxes=rotationAxis(cameraAxes(:,1),-pi/2)*cameraAxes;
    cameraAxes=rotationAxis(cameraAxes(:,1),-cameraAngle)*cameraAxes;
elseif choose==3 %right foot view
    
    cameraPos=pos_data(19,:);%we should add some deviation to avoid problems with the joint
    footAxes=quat2mat((quats(7,:)));
    
    cameraPos=cameraPos-500*footAxes(:,3)';
    cameraAxes=rotationAxis(footAxes(:,3),pi/2)*footAxes;
end

%Change camera to reference segment coordinates
cameraPos=changeCoordinateSystem(cameraPos,head_quat,head_pos);
cameraAxes=inv(quat2mat(head_quat))*cameraAxes;


Rt=inv(cameraAxes); Tt=Rt*(-cameraPos);
img=[];
for i = 1:length(obj_points(:,1))
    p=obj_points(i,:)';
    p=Rt*p+Tt;
%     p=applyDistortion(p,mtx,dist)';
    p=mtx*p;
    p=p./p(end);
    img=[img;p(1),p(2)];
end


scatter(img(:,1),img(:,2),[],'green','filled');
names=mvnx.subject.segments.segment;
for i=[7,11,15]
    text(img(i,1),img(i,2),string(names(i).label));
end

segments=[10,11;9,10;14,15;14,13;19,18;18,17;17,16;16,1;1,20;...
        20,21;21,22;22,23;1,2;2,3;3,4;4,5;5,6;6,7;...
        8,9;13,12;5,12;5,8;12,8;];

for row =segments.'
   p1=img(row(1),:);
   p2=img(row(2),:);
   if inSight(p1,height,width,off) && inSight(p2,height,width,off)
       plot([p1(1),p2(1)],[p1(2),p2(2)],'g');
   end
end

axis([0,1920,0,1080]);


%---------------------------------------------------
%Now lets use the toeric imagepoints and use an PnP algorithm to 
%estimate the R and T of the camera (Rt,Tt are the teoric ones)

image_points=[img,ones(length(img(:,1)),1)]; %add column of ones

if noise~= 0  %add some noise to the imagepoints, for realism
    newimage=zeros(size(image_points));
    newimage(:,3)=ones(length(newimage(:,1)),1);
    for p = 1:length(image_points(:,1))
        angle=rand*2*pi; %random value between 0 and 2pi
        vector=noise*min(height,width)*[cos(angle), sin(angle)];
        newimage(p,[1,2])=image_points(p,[1,2])+vector;
    end
    clear image_points; image_points=newimage;
   scatter(image_points(:,1),image_points(:,2),[],0.5*ones(1,3),'filled');
   for row =segments.'
       p1=image_points(row(1),:);
       p2=image_points(row(2),:);
       if inSight(p1,height,width,off) && inSight(p2,height,width,off)
           plot([p1(1),p2(1)],[p1(2),p2(2)],'color',0.5*ones(3,1),'LineStyle','--');
       end
    end
end
min_error=Inf;
Rs=[]; Ts=[];
for tries=1:100
    if size(Rs,1)==0
        Rini=randrotmat(1);
    else
        Rini=Rs;
    end
    [R,T,rec,num_iter]=Hager(obj_points,image_points,mtx,Rini);

    to_plot=1:length(index);

    imgp=[];
    for idx = to_plot
        p=obj_points(idx,:)';
        p=R*p+T;
    %     p=applyDistortion(p,mtx,dist)';
        p=mtx*p;
        p=p./p(end);
        imgp=[imgp;p(1),p(2)];
    end

    %Calculate error
    error=img-imgp;
    error=error(:,1).^2+error(:,2).^2;
    error=sum(sqrt(error));
    if error<min_error
        clear min_error;
        clear Rs;
        clear Ts;
        min_error=error
        Rs=R;Ts=T;
        if min_error<0.5
            break
        end
    end
end
R=Rs;T=Ts; error=min_error;
scatter(imgp(:,1),imgp(:,2),[],'red','filled');
for row =segments.'
   p1=imgp(row(1),:);
   p2=imgp(row(2),:);
   if inSight(p1,height,width,off) && inSight(p2,height,width,off)
       plot([p1(1),p2(1)],[p1(2),p2(2)],'r');
   end
end

%Show error


text_size=0.02*min(width,height);
t=strcat("Error: ",num2str(error));
text(2*text_size,2*text_size,t);




%Add legend
hold on;
h=zeros(2,1);
h(1)=plot(NaN,NaN,'g');
h(2)=plot(NaN,NaN,'r');
if noise~=0
    h=[h;0];
    h(3)=plot(NaN,NaN,'color',0.5*ones(3,1),'LineStyle','--')
    legend(h,'teoric', 'PnP result','Joints with noise');
else
    legend(h,'teoric', 'PnP result');
end

%Add title
if choose==1
    v="Half Side";
elseif choose==2
    v="Egocentric Camera";
elseif choose==3
    v="Under Right Foot";
else
    v="Undefined";
end
tle=strcat("PnP error at frame ",num2str(vframe)," with ",v," view");

if noise~=0
   tle=strcat(tle," (with ",num2str(100*noise),"% of noise)");
end
title(char(tle));


