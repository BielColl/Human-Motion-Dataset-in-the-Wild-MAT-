function rotMat = rotationAxis(axis,angle)

axis=axis*(1/norm(axis)); %normalize vector
axis=reshape(axis,1,3);

quat=[cos(angle/2), axis*sin(angle/2)];
rotMat=quat2rotmat(quat);

end