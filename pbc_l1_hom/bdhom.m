%  Augmented lagrangian APG-PAS for SVM optimizations.
function [xk,thom,tapg,In_num,Out_num]=bdhom(Q,r,x0,para)
% load('matlab.mat')
% condQ=1e6;% 
% Q=HJJ;
% r=Jf;
% n=size(Q,2);
% x0=zeros(n,1);
% %%
% para.ker='Lin';
% para.IterMax=1000000;
% para.C=10;
% para.CK=100;
% para.d=2;para.c=0;
% para.sigma=0.1;
% para.tolfun=1e-4;
% para.Fmin_permit_ratio=1;
% para.Lambdacho=0;
% %%
% para.tol=1e-4;
% para.tolfactor_res=100;
% para.tolfactor_x=10;
% para.Fmin_permit=4;
% para.delta=0.5e-4;
% if strcmp(para.ker,'Lin')==1    
%     para.speed=2;
%     para.tolfactor_res=100;
% else    
%     para.speed=4;
% end
% para.bet=1e1;
% para.Smax=5;
% para.alh_maxiter=20;
% para.apgtol=1e-4;
% para.trun=2;para.trun_factor=1.03;   %linear kernel 10; Gaussian kerbel 5;
% para.output=1;
% para.condQ=1e6;

%%
C=para.C;
Smax=para.Smax;
n=size(Q,2);
uu=rand(n,1);
uu_uv=1;
while(uu_uv>10^-3)    
    uv=Q*uu;
    Qc=(norm(uv));
    uv=uv/Qc;
    uu_uv=norm(uv-uu);
    uu=uv;
end
L=Qc+1;
vk=x0;
t=1;
errnumx0=2*n;
count=0;
errcha=1;
indfix=0;
time=tic;
while((errcha>para.apgtol|| indfix<Smax))   
    tk=(1+sqrt(1+4*t^2))/2;
    if (sum(vk~=0))/n>0.4
        gk=Q*vk+r;
    else
        gk=Q*sparse(vk)+r;
    end
    a=(vk-1/L*gk);
    xk=a.*(a>0&a<C)+C*(a>=C);
    vk=xk+((t-1)/tk)*(xk-x0);
    normxk=norm(xk);
    errcha=norm(x0-xk)/normxk;
    errnumxk=sum(xk>para.apgtol*normxk);
    errnum=abs(errnumx0-errnumxk);
    idi=(errnum==0);
    indfix=(indfix+idi)*idi;
    errnumx0=errnumxk;
    t=tk;
    count=count+1;
     x0=xk;
end
tapg=toc(time);
trunxk=norm(xk)*para.apgtol/para.trun;
Jl=find(xk<trunxk);
Jm=find(trunxk<xk&xk<C-trunxk);
Ju=find(xk>C-trunxk);
xinit=zeros(n,1);
xinit(Jm)=xk(Jm);
xinit(Ju)=C;
res=Q*xinit+r; %  res=Q(:,J)*xk(J)+r;
w=zeros(n,1);
w(Jm)=-res(Jm);
w(Jl)=-min(res(Jl))+10;
w(Ju)=-max(res(Ju))-10;
u=r+w;
time=tic;
[xk,In_num,Out_num]=homobound_epscorr(Q,xinit,u,-w,Jl,Jm,Ju,C,para.condQ);
thom=toc(time);




