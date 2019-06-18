framesFolder='Frames';mtxFile=mvnx;video_time_cut=[2,5];data_time_cut=[0,5];


close all;
frames_cut=time_cut.*60;
f1=figure();
frame=1;
fld=convertCharsToStrings(framesFolder);

indexes=frames_cut(1):frames_cut(2);
total=length(indexes);
fprintf("Uploading frames \n")


if exist('images','var')~=1
    images={};
    for i = 1: length(indexes)

        imagename=char(strcat(fld,"/","frame_",num2str(indexes(i)),".jpg"));
        image=imread(imagename);
        image=flip(image,1);
        image=flip(image,2);
        images{i}=image;
    end
end

claps_video=[];
done=true;
while true
    
    imshow(images{frame});
    hold on;
    text(50,50,strcat("Frame: ",num2str(indexes(frame))), 'color',[1,1,1]);
    text(50,100,strcat("Claps frames: ",num2str(claps_video)), 'color',[1,1,1]);
    w=waitforbuttonpress;
    if w==1
        k=f1.CurrentCharacter;
        if k=='q'
            close(f1);
            if length(claps_video)==0
                done=false;
            end
            break;
        elseif k=='d'
            if frame~=total
                frame=frame+1;
            else
                frame=1;
            end
            continue
        elseif k=='a'
            if frame~=1
                frame=frame-1;
            else
                frame=total;
            end
            continue
        elseif k=='m'
            claps_video=[claps_video, indexes(frame)];
        elseif k=='n'
            claps_video=claps_video(1:end-1);
   
       end
    end
    pause(1/60); hold off;
end
if done
    fprintf(strcat("Claps found in video frames ",num2str(claps_video),"\n"))
    total_claps=length(claps_video);
    nframes=length(mvnx.subject.frames.frame);

    rightHandAcc=[];
    indexes=data_time_cut.*60; indexes=indexes(1):indexes(2); indexes=indexes+4;
    for i=indexes
        pos_data = mvnx.subject.frames.frame(i).acceleration;
        pos_data=reshape(pos_data,3,length(pos_data)/3)';

        rightHandAcc=[rightHandAcc;pos_data(11,:)];
    end

    t=0:length(rightHandAcc(:,1))-1;
    f3=figure();
    data_claps=[];
    for i =1:length(rightHandAcc(1,:))
        plot(t,rightHandAcc(:,i));
        peakh=max(rightHandAcc(:,1))-0.0001;
        [a,b]=findpeaks(rightHandAcc(:,1),'MinPeakDistance',10,'MinPeakHeight',peakh);
        while length(b)~=total_claps
            peakh=peakh*0.95;
            [a,b]=findpeaks(rightHandAcc(:,1),'MinPeakDistance',10,'MinPeakHeight',peakh);
        end
        data_claps=[data_claps;b'];
        hold on
    end
    claps_data=sum(data_claps);
    claps_data=claps_data./length(rightHandAcc(1,:));
    claps_data=round(claps_data);
    grid on;
    x_claps=claps_data;
    y_claps=[-50,60];


    title('Finding claps frames through the right hand acceleration')
    ylabel('Segment acceleration [m/s^2]')
    xlabel('Data frames')
    xt=[t(1):20:t(end),x_claps]; xt=sort(xt); xticks(xt);

    legend('X-Acceleration','Y-Acceleration','Z-Acceleration', 'Location','northwest');
    axis equal;
    clap1=find(xt==x_claps(1));
    clap2=find(xt==x_claps(2));
    axis([0,250,y_claps])
    xlab=strings(1,length(xt));
    xlab(clap1)=num2str(xt(clap1));
    xlab(clap2)=num2str(xt(clap2));

    xticklabels(num2cell(xlab))
    line([x_claps(1),x_claps(1)],y_claps,'LineStyle','--', 'HandleVisibility','off');
    line([x_claps(2),x_claps(2)],y_claps,'LineStyle','--','HandleVisibility','off');


    fprintf(strcat("Claps found in data frames ",num2str(claps_data),"\n"))

    sync=round(mean(claps_video-claps_data));

    fprintf(strcat("Sync value: ",num2str(sync),"\n"))


end