function xk=subFISTA(A,y,xinit,para)
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
%%
L=para.L;lamb=para.lamb;
n=length(xinit);
x0=xinit;vk=x0;
% itermax=n;
err=1;
iter=1;
%%  低精度解

while( iter<para.itermax)  
    
    if para.chose==1
        Avk=A*vk;
        gk=A'*(Avk-y);
        fvalue=1/2*norm(Avk-y)^2+para.lamb*sum(abs(vk));
    else
        gk=A*vk+y;
        
        fvalue=1/2*((gk+y)'*vk)+para.lamb*sum(abs(vk));
    end
    if iter==1
        fvalue0=fvalue;
    else
        if fvalue>fvalue0
            L=1.1*L;
            fvalue0=fvalue;
        end
    end
    a=(vk-1/L*gk);
    xk=(a-sign(a)*lamb/L).*(abs(a)>(lamb/L));
    vk=((2*iter+1)/(iter+2))*xk+((1-iter)/(iter+2))*x0;
    if (sum(vk~=0))/n<0.4
        vk=sparse(vk);
    end
    err=norm(x0-xk)/(norm(xk)+0.00001);
    x0=xk;
    iter=iter+1;
end
