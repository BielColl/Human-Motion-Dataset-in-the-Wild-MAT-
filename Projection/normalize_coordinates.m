function Unorm = normalize_coordinates(U,A)

fu=A(1,1);
fv=A(2,2);
u0=A(1,3);
v0=A(2,3);

Unorm=zeros(size(U));
Unorm(:,1)=(U(:,1)-u0)/fu;
Unorm(:,2)=(U(:,2)-v0)/fv;
