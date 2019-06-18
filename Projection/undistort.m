function new_image = undistort(image,mtx,dist)

cameraParam=cameraParameters("IntrinsicMatrix",mtx',"RadialDistortion",dist([1,2]),"TangentialDistortion",dist([3,4])) ;
[J,newO]=undistortImage(image,cameraParam,'OutputView','same');
new_image=J;

end