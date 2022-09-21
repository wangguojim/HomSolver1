function x=lasso_apg(A,y,xinit,para)
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
vk=xinit;
x0=vk;err=1;count=0;
%%  APG求低精度解
while(err>para.apgtol&&count<para.countmax)
    count=count+1;
    gk=A'*(A*vk-y);
    a=(vk-1/L*gk);
    xk=(a-sign(a)*lamb/L).*(abs(a)>(lamb/L));
    vk=sparse(((2*count+1)/(count+2))*xk+((1-count)/(count+2))*x0);
    err=norm(x0-xk)/norm(xk);
    x0=xk;
end
x=xk;