% Pose Projection Interactive App
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
vframe=951;
dframe=vframe-sync+3;
test_frame=[160:2:250, 650:2:900];
skip=2; fps=60/skip;
gif_frames=160:skip:1020;
stamp=datestr(clock);
stamp=strrep(stamp,'-','_');
stamp=strrep(stamp,':','_');
stamp=strrep(stamp,' ','_');
gif_name=strcat("projection_",stamp);
max_iter=300; noise=0.03;


%Initial plot of image
f1=figure;
imagename=strcat('Frames To Start/frame_',int2str(vframe),'.jpg');
image=imread(imagename);
image=flip(image,1);
image=flip(image,2);
radial_dist=dist([1,2]);
tang_dist=dist([3,4]);
cameraParams=cameraParameters('IntrinsicMatrix',mtx','RadialDistortion',radial_dist,...
    'TangentialDistortion',tang_dist);
% image=undistortImage(image,cameraParams);
s=size(image); height=s(1);width=s(2); clear s;
imshow(image);
mtx(1,3)=width/2; mtx(2,3)=height/2;


%Select initial imagepoints
avail_segments=["right wrist", "right elbow", "left wrist", "left elbow",...
                "right toe", "right ankle", "right knee", "right hip",...
                "left toe", "left ankle", "left knee", "left hip", "hip"];

selected=1:length(avail_segments);
if vframe==290
    selected=[1,2,5,9,10,11];
elseif vframe==124
    selected=[1,2,3,4,5,6,9,8,12,13];
elseif vframe==627
    selected=[1,2,3,4,5,7,8,9,12,13];
elseif vframe==647
    selected=[1,2,4,5,9,6,7,8,12,13];
elseif vframe==951
    selected=[1,2,3,5,9,7,11,8];
elseif vframe==759 || vframe==740
    selected=[1,3,4,5,9,7,11];
elseif vframe==702
    selected=[1,3,4,5,9,7,11,8,12,13];
end

pc=containers.Map;
hold on;
finished=false;
labels=[];
for s =selected
    segment_name=avail_segments(s);
    labels=[labels,segment_name];
    mess=strcat("Adding imgage point of ", segment_name);
    htext=text(0,50,mess,'Color','white');
    point=ginput(1);
    plot(point(1),point(2),'r+');
    pc(char(segment_name))=point;
    pause(0.2);
    delete(htext);

end

%Importing MVNX Data and reshaping it
pos_data = mvnx.subject.frames.frame(dframe).position;
pos_data=reshape(pos_data,3,length(pos_data)/3)';
pos_data=pos_data*1000;

%New world coordinate system
ref_segment=24;
head_pos=pos_data(ref_segment,:);
quats=mvnx.subject.frames.frame(dframe).orientation;
quats=reshape(quats,4,length(quats)/4)';
head_quat=quats(ref_segment,:);

%Creating image_points and object_points in new coordinate system
prop_quat=mvnx.subject.frames.frame(dframe).orientation(end-3:end);
index=[];
segmentIndex;
img_points=[];
for i = labels
   index=[index,segIndex(char(i))];
   p=pc(char(i));
   img_points=[img_points;p(1),p(2),1];
end
obj_points=[];

for i =index
   row=pos_data(i,:);
   p=changeCoordinateSystem(row,head_quat,head_pos);
   obj_points=[obj_points;reshape(p,1,3)];
end

%------------------------------------
%PRINCIPAL APP LOOP
%------------------------------------

calculated=false;
clear error; clear cp_img_points;



while ~finished
    if ~calculated  %just recalculate if told to do so
        min_error=Inf;
        if exist('error','var')==1 %for further iterations
            min_error=error;
        end
        Rs=[]; Ts=[];
        if exist('cp_img_points','var')~=1 %if copy not exists
           cp_img_points=img_points; 
        else
           img_points=cp_img_points; %cp_img_points is later used to save the original img_points
        end    
        best_image=img_points; %this variable saves the image points with best results
        
        for tries=1:max_iter
            if size(Rs,1)==0
                Rini=randrotmat(1);
            else
                if error>50
                    Rini=randrotmat(1);
                else
                    Rini=Rs;
                end
            end
            if tries<0.5*max_iter
                [R,T,rec,num_iter]=Hager(obj_points,cp_img_points,mtx,Rini);
            else
                noised=addNoise2Points(img_points(:,[1,2]),noise*min(width,height));
                cp_img_points=[noised,ones(length(noised(:,1)),1)];
                [R,T,rec,num_iter]=Hager(obj_points,cp_img_points,mtx,Rini);
            end

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
            error=cp_img_points(:,[1,2])-imgp;
            error=error(:,1).^2+error(:,2).^2;
            error=sum(sqrt(error))/length(img_points(:,1));
            if error<min_error && size(R,1)==3 && size(R,2)==3
                clear min_error;
                clear Rs;
                clear Ts;
                min_error=error;
                if tries>0.5*max_iter
                   best_image=cp_img_points;
                end
                Rs=R;Ts=T;
                if min_error<0.5
                    break
                end
            end
        end
        calculated=true;
        cp_img_points=img_points;
        img_points=best_image;
        if size(Rs,1)==3 && size(Rs,2)==3
            R=Rs;T=Ts; 
        else
            R=bestR; T=bestT; %this happens when no better solution than the previous pass is found
            fprintf('No better solution found \n');
        end
        error=min_error;
    end
    
    % [R,T]=OPnP(obj_points',img_points');
    %Plot projections with R and T to check whether it works or not
    hold on
    to_plot=1:length(index);
    sse=0;
    for idx = to_plot
        p=obj_points(idx,:)';
        p=R*p+T;
        %p=applyDistortion(p,mtx,dist)';
        p=mtx*p;
        p=p./p(end);


        segment_name=labels(idx);
        o_img=pc(char(segment_name));
        img=img_points(idx,:);
        mp=(p(1:2)+img(1:2)')./2;
        sse=sse+((img(1)-p(1))^2+(img(2)-p(2))^2);
        text(mp(1),mp(2), labels(idx), 'Color', 'white');

        plot([p(1),img(1)],[p(2),img(2)],'g');
        scatter(p(1),p(2),[],'blue','filled');
        plot(img(1),img(2),'r+');
        plot(o_img(1),o_img(2),'g+');
        pause(0.05);
    end
    text(50,50,strcat("Error: ",num2str(error)),'Color','white');
    text(50,100,strcat("Nr of iter:", num2str(tries)),'Color','white');
    
    if tries >0.5*max_iter %if noise has been used
        %Plot a circle that shows the noise used
        for p =1:length(cp_img_points(:,1))
            cx=cp_img_points(p,1);
            cy=cp_img_points(p,2);
            radius=noise*min(width,height);
            angles=linspace(0,2*pi,200);
            xunit=radius*cos(angles)+cx;
            yunit=radius*sin(angles)+cy;
            plot(xunit,yunit,'color',ones(3,1));
        end
    end
    f2=knownProjection(test_frame,sync,R,T,mvnx,ref_segment,mtx,1);
     
    
    close(f2);
    figure(f1);
    bestR=R; bestT=T;
    w=waitforbuttonpress;
    if w==1 && f1.CurrentCharacter=='q'
        close(f1);
        return
    elseif w==1 && f1.CurrentCharacter=='r'
        finished=false;
        calculated=false;
        close(f1); 
        f1=figure();
        imshow(image);
        Rs=R;Ts=T;
    elseif w==1 && f1.CurrentCharacter=='p'
        createProjectionsMP4(char(gif_name),gif_frames,sync,R,T,mvnx,ref_segment,mtx,fps)
        close(f1);
        return
    elseif w==1 && f1.CurrentCharacter=='e'
        %edit one of the imagepoints
        finished=false;
        calculated=false;
        avalKeys=["a","s","d","f","g","h","j","k","l","z","x","c","v","b",...
            "n","m"];
        keys=avalKeys(1:length(labels));
        for i = 1:length(labels)
            seg=labels(i);
            k=keys(i);
            t=strcat("Press ",k," to edit ", seg);
            text_size=0.01*min(height,width);
            ipos=200;
            figure(f1);
            text(200,ipos+(i*2.5)*text_size,t,'FontSize',ceil(text_size),'Color','white')
        end
        a=waitforbuttonpress;
        if a==1
            pressed=f1.CurrentCharacter;
            k=find(keys==pressed);
            sk=size(k);
            if sk(2)==0
                return
            else
                sel=labels(k);
                point=ginput(1);
                pc(char(sel))=point;
                cp_img_points(k,1)=point(1);
                cp_img_points(k,2)=point(2);
                close(f1); 
                f1=figure();
                imshow(image);
            end
        end
    else
        finished=true;
    end
end
    





