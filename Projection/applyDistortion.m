function point = applyDistortion(p,mtx,dist)

% p should already be in format [u,v,1], but just in case:
%p=p./p(end);
u=p(1); v=p(2);
%unpack distortion parameters and image centers
k1=dist(1);k2=dist(2);p1=dist(3);p2=dist(4);k3=dist(5);
cx=mtx(1,end);cy=mtx(2,end);

%apply distortion
r=sqrt(u^2+v^2);
factor1=1+k1*r^2+k2*r^4+k3*r^6;
factor2u=2*p1*u*v + p2*(r^2+2*u^2);
factor2v=2*p2*u*v + p1*(r^2+2*v^2);

point=[u*factor1+factor2u,v*factor1+factor2v,1];
end