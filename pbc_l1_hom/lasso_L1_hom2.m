function x=lasso_L1_hom2(A,y,w,r,xinit,xpre,para)
%该函数通过参数积极集方法求解
%min 1/2*||Ax-b||^2+lamb*||x||_1+t*r'*x+w'*x
%t从零跟踪到1，即求得
%min 1/2*||Ax-b||^2+lamb*||x||_1
%% parameters initilization
delta=para.delta;
lamb=para.lamb;
x=xinit;
n=length(xinit);
epsilon=10^-10;   %设置为10^-8效果较好
tmax=1;
%%
Jc=find(abs(x)==0);
J=find(x~=0);
% r=-w;
%%
tj=0;
count=1;
trecord=0;
Ay=A'*y;
while(tj<tmax)
    count        = count+1;
    AJ=A(:,J);
    QJJ=AJ'*AJ+delta*speye(length(J));
    signxJ       = sign(AJ'*(y-AJ*x(J))-delta*(x(J)-xpre(J))-w(J)-tj*r(J));
    if ~isempty(J)
        uv=QJJ\([AJ'*y-w(J)+delta*xpre(J)-lamb*signxJ,-r(J)]);
        u=uv(:,1); v=uv(:,2);   %x=u+vt
        a=A'*(AJ*u-y)+w-delta*xpre;
        b=A'*(AJ*v)+r;
    else
        u=[];v=[];
        a=A'*(-y)+w-delta*xpre;
        b=r;
    end
    % find the additional index
  
    a=a(Jc);b=b(Jc);  %res=a+tb
    %% -------------------------------------------------------%%
    tallx=-u./v;
    tupx=find(tallx>trecord(end)+10*epsilon);
    [tjx,idx]=min(tallx(tupx)) ;
    idx=tupx(idx);
    tallres=-(a-lamb*sign(b))./b;
    tupres=find(tallres>trecord(end)+10*epsilon);
    [tjres,idres]=min(tallres(tupres)) ;
    idres=tupres(idres);
    if isempty(tjx); tjx=100; end
    if isempty(tjres); tjres=101; end
    [tj,id]=min([tjx,tjres]);
    %% -----------------------------------------------------------%
    % flash the active set
    if(tj>=1)
        tj=1;
        x(J)          = u+tj*v ;
    else
        x(J)          = u+tj*v ;
        if (id==1)
            x(J(idx))=0;
            J=setdiff(J,J(idx));
        else if (id==2)
                J(end+1) =Jc(idres);
            end
        end
    end
    Jc=setdiff(1:n,J);
    trecord(count) = tj; 
end
