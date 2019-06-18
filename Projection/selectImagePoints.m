function [img_points]=selectImagePoints(bank)

image=bank('image');
imshow(image);
hold on;

selected=bank('selected');
tags=bank('tags');

pc=containers.Map;
img_points=[];
for i=1:length(selected)
    segment_name=tags(i);
    mess=strcat("Adding imgage point of ", segment_name);
    htext=text(0,50,mess,'Color','white');
    point=ginput(1);
    plot(point(1),point(2),'r+');
    pc(char(segment_name))=point;
    img_points=[img_points;point,1];
    pause(0.2);
    delete(htext);
end

bank('pc')=pc;
end