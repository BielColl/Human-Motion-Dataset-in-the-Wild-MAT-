pc=containers.Map;
pc('right wrist')=[667,375]*2;
pc('right elbow')=[674,498]*2;
pc('left wrist')=[41,72]*2;
pc('left elbow')=[79,259]*2;
pc('right toe')=[577,192]*2;
pc('right ankle')=[539,243]*2;
pc('right knee')=[557,241]*2;
pc('right hip')=[570,325]*2;
pc('left toe')=[449,177]*2;
pc('left ankle')=[457,242]*2;
pc('left knee')=[442,234]*2;
pc('left hip')=[406,282]*2;
pc('hip')=[477,304]*2;


labels=string(pc.keys);
img_points=cell2mat(pc.values');
img_points=[img_points,ones(length(img_points(:,1)),1)];

pos_data = mvnx.subject.frames.frame(frame).position;
pos_data=reshape(pos_data,3,length(pos_data)/3)';
pos_data=pos_data*1000;

ref_segment=7;
head_pos=pos_data(ref_segment,:);
quats=mvnx.subject.frames.frame(frame).orientation;
quats=reshape(quats,4,length(quats)/4)';
head_quat=quats(ref_segment,:);

prop_quat=mvnx.subject.frames.frame(frame).orientation(end-3:end);
index=[];
segmentIndex;
for i = labels
   index=[index,segIndex(char(i))];
end
obj_points=[];
for i =index
   row=pos_data(i,:);
   p=changeCoordinateSystem(row,head_quat,head_pos);
   obj_points=[obj_points;reshape(p,1,3)];
end


save('projection.mat','obj_points','img_points','mtx');