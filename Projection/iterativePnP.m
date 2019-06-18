function [R,T,min_error]=iterativePnP(img_points,obj_points,mtx,noise,min_error)

finished=false;
if ~exist('min_error','var')
    min_error=Inf;
end

if ~exist('noise','var')
    noise=1;
end
max_iter=400;

Rs=[]; Ts=[];
if exist('cp_img_points','var')~=1 %if copy not exists
   cp_img_points=img_points; 
else
   img_points=cp_img_points; %cp_img_points is later used to save the original img_points
end    
best_image=img_points; %this variable saves the image points with best results

for tries=1:max_iter

    %Obtain some R and T
    if size(Rs,1)==0
        Rini=randrotmat(1);
    else
        if sse>50
            Rini=randrotmat(1);
        else
            Rini=Rs;
        end
    end
    if tries<0.5*max_iter
        [R,T,rec,num_iter]=Hager(obj_points,cp_img_points,mtx,Rini);
    else
        noised=addNoise2Points(img_points(:,[1,2]),noise);
        cp_img_points=[noised,ones(length(noised(:,1)),1)];
        [R,T,rec,num_iter]=Hager(obj_points,cp_img_points,mtx,Rini);
    end

    %Calculate its sse

    %First project obj_points
    imgp=[];
    for i=1:length(obj_points(:,1))
        p=obj_points(i,:)';
        p=R*p+T;
    %     p=applyDistortion(p,mtx,dist)';
        p=mtx*p;
        p=p./p(end);
        imgp=[imgp;p(1),p(2)];  
    end

    sse=cp_img_points(:,[1,2])-imgp;
    sse=sse(:,1).^2+sse(:,2).^2;
    sse=sum(sqrt(sse))/length(img_points(:,1));
    if sse<min_error && size(R,1)==3 && size(R,2)==3
        clear min_error;
        clear Rs;
        clear Ts;
        min_error=sse;
        if tries>0.5*max_iter
           best_image=cp_img_points;
        end
        Rs=R;Ts=T;
        if min_error<0.5
            break
        end
    end
end

end