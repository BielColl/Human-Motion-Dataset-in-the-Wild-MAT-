function [obj_points]=obtainObjPoints(mvnx,bank, ref_segment)

dframe=bank('dframe');
%Importing MVNX Data and reshaping it
pos_data = mvnx.subject.frames.frame(dframe).position;
pos_data=reshape(pos_data,3,length(pos_data)/3)';
pos_data=pos_data*1000;

%New world coordinate system
head_pos=pos_data(ref_segment,:);
quats=mvnx.subject.frames.frame(dframe).orientation;
quats=reshape(quats,4,length(quats)/4)';
head_quat=quats(ref_segment,:);

segIndex=containers.Map;
segIndex('right wrist')=11;
segIndex('right elbow')=10;
segIndex('left wrist')=15;
segIndex('left elbow')=14;
segIndex('right toe')=19;
segIndex('right ankle')=18;
segIndex('right knee')=17;
segIndex('right hip')=16;
segIndex('left toe')=23;
segIndex('left ankle')=22;
segIndex('left knee')=21;
segIndex('left hip')=20;
segIndex('hip')=1;

segments=[];
for label=bank('tags')
    segments=[segments,segIndex(char(label))];
end
obj_points=[];
for i=segments
   row=pos_data(i,:);
   p=changeCoordinateSystem(row,head_quat,head_pos);
   obj_points=[obj_points;reshape(p,1,3)];
end
end