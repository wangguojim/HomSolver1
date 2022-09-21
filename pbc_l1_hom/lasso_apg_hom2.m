function x=lasso_apg_hom2(Q,f,xinit,para)
%%
%APG-Homotopy算法求解Lasso问题
%min 1/2x'Qx+f'x+lamb*||x||_1
%%
n=length(f);


trunfactor=1;
lamb=para.lamb;

uu_uv=1;uu=rand(n,1);
while(uu_uv>10^-2)
    uv=Q*uu;Ac=norm(uv);uv=uv/Ac;uu_uv=norm(uv-uu);uu=uv;
end
L=Ac+5;
vk=xinit;
x0=vk;err=1;count=0;
%%  APG求低精度解
while(err>para.apgtol&&count<para.countmax)
    count=count+1;
    gk=Q*vk+f;
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
xi=zeros(n,1);
xi(J)=xk(J);
%%  homotopy求精确解
% x2=lasso_L1_hom1_cp(Q,f,x,x,para);% norm(xx-x1)
x=Lasso_L1_hom3(Q,f,xi,para);% norm(xx-x1)
