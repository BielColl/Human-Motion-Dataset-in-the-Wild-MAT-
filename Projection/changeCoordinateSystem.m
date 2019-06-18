function p3d = changeCoordinateSystem(p,quat,origin)
% origin should be the position of the new coordinate systema in the old
% coordinate system
%quat should be the orientation of the new coordinate system in the old one
%p is the point we want to redefine

rotmat=quat2mat(quat);
R=inv(rotmat);
T=-origin;

p=reshape(p,3,1); %just in case
T=reshape(T,3,1);
p3d=R*(p+T);

end