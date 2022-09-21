function xk=ISTA(A,y,lamb,L)
%ISTA算法求解Lasso问题
%min 1/2||Ax-y||^2+lamb*||x||_1

%% ----------------------testing-----------------------
% clc;
% K=6;
% m=2^K;
% n=2^(K+4);
% lamb=0.1;
% A=randn(m,n);
% y=randn(m,1);
% Ay=A'*y;
% eig=svd(A);
% L=(max(eig))^2;
%%
[m,n]=size(A);
xk=zeros(n,1);
itermax=n;
errtol=1e-4;err=1;
iter=1;
%%
while(err>errtol&&iter<itermax)   
    x0=xk;
    gk=A'*(A*xk-y);
    a=(xk-1/L*gk);
    xk=(a-sign(a)*lamb/L).*(abs(a)>(lamb/L));
    err=norm(x0-xk)/norm(xk);   
    iter=iter+1;
end
  