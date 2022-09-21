function xk=qp_L1_hom(Q,r,xinit,para)
% tic;
% m=1000;n=2000;
% A=randn(m,n);
% Q=A'*A;
% r=A'*randn(m,1);
% xinit=zeros(n,1);
% 
% para.funtol1=1e-16;
% para.lamb=1e0

f=@(x)(1/2*x'*Q*x+r'*x+para.lamb*sum(abs(x)));
%%
CK=para.CK2;
xk=xinit;
n=length(r);
res=Q*xk+r;
fvalue=f(xk);
reswhole=res+para.lamb*sign(xk);
W1=find(xk==0&abs(res)>(para.lamb));    W2=find(xk~=0&abs(reswhole)>0);
iter=1;
% Q=full(Q);
while (iter<para.inner_iter_max)
    W=[];
    resW1=res(W1); resW2=reswhole(W2);
    [~,W1sort]=sort(abs(resW1));[~,W2sort]=sort(abs(resW2));    
    W=union(W,W1(W1sort(end-min(length(W1),CK)+1:end)));
    W=union(W,W2(W2sort(end-min(length(W2),CK)+1:end))); 
    lenW=length(W);    
    QWW=Q(W,W);
    
    if mod(iter,10)==1
        para.delta=max_svdnum(QWW,1)/10^7;
    end    
    %% FISTAÇ°¼ÓproximalÏî
    ys=res(W)-QWW*xk(W)-para.delta*xk(W);
    Qk=QWW+para.delta*speye(length(W));
    u=lasso_apg_hom2(Qk,ys,xk(W),para);
    value_drese=1/2*u'*(Qk*u)+ys'*u+para.lamb*sum(abs(u))-(1/2*xk(W)'*(Qk*xk(W))+ys'*xk(W)+para.lamb*sum(abs(xk(W))))-1/2*para.delta*norm(xk(W)-u)^2;
    %%
    fvalue=fvalue+value_drese;
    value_drese;
    if  value_drese>0
%         fprintf('subsubproblem solved low-preccionly\n')
    end
    h=sparse(n,1);
    h(W)=u-xk(W);
    xk(W)=u;
    res=Q*h+res;% res=Q*xk+r;
    reswhole=res+para.lamb*sign(xk);
    W1=find(xk==0&abs(res)>(para.lamb));    W2=find(xk~=0&abs(reswhole)>0);    
    if abs(value_drese/(0.001+abs(fvalue)))<para.funtol1
        break;
    end
    iter=iter+1;
end
% kt=@(x) (Q*x+r);kt1=@(x)(abs(kt(x))-para.lamb);kkt=@(x) (x~=0).*(kt(x)+para.lamb*sign(x))+kt1(x).*(kt1(x)>0).*(x==0);

