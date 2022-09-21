function x=lasso_apg_hom(A,y,xinit,para)
%%
%APG-Homotopy算法求解Lasso问题
%min 1/2||Ax-y||^2+lamb*||x||_1
%%
trunfactor=5;
lamb=para.lamb;
[m,n]=size(A);
uu_uv=1;uu=rand(n,1);
while(uu_uv>10^-2)
    uv=A'*(A*uu);Ac=norm(uv);uv=uv/Ac;uu_uv=norm(uv-uu);uu=uv;
end
L=Ac+5;
para.L=L;
vk=xinit;
x0=vk;err=1;count=0;
%%  APG求低精度解
while(err>para.apgtol&&count<para.countmax)
    count=count+1;   
    gk=A'*(A*vk-y);
    a=(vk-1/L*gk);
    xk=(a-sign(a)*lamb/L).*(abs(a)>(lamb/L));
    vk=((2*count+1)/(count+2))*xk+((1-count)/(count+2))*x0;
    if sum(vk~=0)/n<0.4
        vk=sparse(vk);
    end
    err=norm(x0-xk)/norm(xk);
    x0=xk;
end

J=find(abs(xk)>norm(xk)*trunfactor*para.apgtol);
x=zeros(n,1);
x(J)=xk(J);
%%  homotopy求精确解
x=lasso_L1_hom(A,y,x,x,para);%  

%%
% xinit=xk;xpre=xk;
% delta=para.delta;
% lamb=para.lamb;
% x=xinit;
% n=length(xinit);
% epsilon=10^-10;   %设置为10^-8效果较好
% tmax=1;
% %%
% Jc=find(abs(x)<norm(x)*10^-5);
% x(Jc)=0;
% F=A'*(A*x-y)+delta*(x-xpre);
% w=-F-lamb*sign(x);
% J=find(x~=0);
% if ~isempty(Jc)
%     w(Jc)=(F(Jc))/max(abs(F(Jc)))*(lamb-0.2)-F(Jc);
% end
% r=-w;
% para.apgtol=para.apgtol/10;
% T=2;ht=1/T;
% x11=xk;
% for i=1:T
%     x11=lasso_apg_hom1(A,y,w+i*ht*r,x11,para);
%     xpre=x11;
% end
% norm(x-x11)
% 1/2*norm(A*x-y)^2+para.lamb*sum(abs(x))
% 1/2*norm(A*x11-y)^2+para.lamb*sum(abs(x11))