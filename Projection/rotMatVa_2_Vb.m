function rotMat=rotMatVa_2_Vb(va,vb)
%Function to find the rotation matrix for going from vector A to vector B

%Rotation axis is the perpendicular to both of them

rv=cross(va,vb);

%Angle of rotation
theta=acos(dot(va,vb)/(norm(va)*norm(vb)));

rotMat=rotationAxis(rv,theta);


end