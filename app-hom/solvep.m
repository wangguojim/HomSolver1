function X=solvep1(R,r) %该方程求解R'*Rx=r,其中R是上三角
Rrow  = size(R,2);
Y=r/R(1,1);
for i=2:Rrow
    Y(i,:)=(r(i,:)-R(1:i-1,i)'*Y(1:i-1,:))/R(i,i);
end
X=Y/R(Rrow,Rrow);
for i=2:Rrow
    X(Rrow-i+1,:)=(Y(Rrow-i+1,:)-R(Rrow-i+1,Rrow-i+2:Rrow)*X(Rrow-i+2:Rrow,:))/R(Rrow-i+1,Rrow-i+1);
end