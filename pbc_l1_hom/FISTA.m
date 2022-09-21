function [xk,FFi,TFi]=FISTA(A,y,lamb,L,errtol,itermax)
%FISTA算法求解Lasso问题
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
% lamb=para.lamb;
% errtol=1e-5;
% itermax=500;
%%
[m,n]=size(A);
x0=zeros(n,1);vk=x0;
% itermax=n;
err=1;
iter=1;
time=tic;
%%  低精度解
TFi=0;
FFi=1/2*norm(y)^2;
while(err>errtol&&iter<itermax)
    res=(A*vk-y);
     kt = (A'*res);
    kt = (abs(kt )- lamb);
    kkt= (vk~=0).*(kt + lamb*sign(vk))+kt1 .*(kt1 >0).*(vk==0);
    gk=A'*res;
    if mod(iter,10)==0
        TFi(end+1)=toc(time);
        FFi(end+1)=1/2*norm(res)^2+lamb*sum(abs(vk));
    end
    a=(vk-1/L*gk);
    xk=(a-sign(a)*lamb/L).*(abs(a)>(lamb/L));
    vk=sparse(((2*iter+1)/(iter+2))*xk+((1-iter)/(iter+2))*x0);
    err=norm(x0-xk)/norm(xk)
    x0=xk;
    iter=iter+1;
end
