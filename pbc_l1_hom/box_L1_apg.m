function x=box_L1_apg(Q,r,xinit,ld,ud,para)
%%
% n=100;Q=rand(n,n);Q=Q'*Q;r=randn(n,1);ld=-0.01*ones(n,1);ud=0.01*ones(n,1);xinit=zeros(n,1);
% para.countmax=100000;para.apgtol=1e-4;para.lambda=0.001;
%APG算法求解Lasso问题
%min 1/2||Ax-y||^2+lambda*||x||_1  s.t. ld<=x<=ud
%%

lambda=para.lambda;
n=size(Q,1);
uu_uv=1;uu=rand(n,1);
while(uu_uv>10^-2)
    uv=Q*uu;Ac=norm(uv);uv=uv/Ac;uu_uv=norm(uv-uu);uu=uv;
end
L=Ac+5;
vk=xinit;
x0=vk;err=1;count=0;
t=1;
%%  APG求低精度解
while(err>para.apgtol&&count<para.countmax)
    count=count+1;
    gk=Q*vk+r;
    a=(vk-1/L*gk);
    xk=(a-sign(a)*lambda/L).*(abs(a)>(lambda/L));
    xk=ld.*(xk<=ld)+xk.*(xk>ld&xk<ud)+ud.*(xk>=ud);
     tk=(1+sqrt(1+4*t^2))/2;
%     vk=sparse(((2*count+1)/(count+2))*xk+((1-count)/(count+2))*x0);
    vk=(xk+((t-1)/tk)*(xk-x0));
    err=norm(x0-xk)/norm(xk);
    x0=xk;
     t=tk;
end
x=xk;


