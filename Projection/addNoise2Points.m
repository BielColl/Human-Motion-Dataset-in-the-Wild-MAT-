function np = addNoise2Points(points,noise)
%Function that adds noise to a set of 2D points
%Give the noise as the maximum length a point would be able to move
%Each row of points matrix is a 2D point
n=points;
for i = 1:length(n(:,1))
    angle=rand*2*pi; %random value between 0 and 2pi
    l=rand*noise;
    vector=l*[cos(angle), sin(angle)];
    n(i,:)=n(i,:)+vector;
end
np=n;
end