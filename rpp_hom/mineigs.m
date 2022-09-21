

function [min_eig,maxabs_eig]=mineigs(M,tol)
n=size(M,2);
uu=rand(n,1);
[maxabs_eig,uv]=power_maxeig(M,uu,tol*10);
if maxabs_eig>0
    H=M-(maxabs_eig+0.01)*speye(n);
    min_eig=power_maxeig(H,uu,tol);
    min_eig=min_eig+maxabs_eig+0.01;
else
    min_eig=power_maxeig(M,uv,tol);
end
end


