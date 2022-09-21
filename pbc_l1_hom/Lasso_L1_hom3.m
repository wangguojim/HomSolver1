
function x=Lasso_L1_hom3(Q,f,xinit,para)
%该函数通过参数积极集方从xinit处求解
%min 1/2*x'Qx+f'x+lamb*||x||_1
% 构造同伦
%min 1/2*x'Qx+f'x+lamb*||x||_1+t*w'*x+w'*x
% t从零跟踪到1，即求得
%% parameters initilization
lamb=para.lamb;
x=xinit;
n=length(xinit);
epsilon=10^-10;   %设置为10^-8效果较好
tmax=1;
%%
Jc=find(abs(x)==0);
x(Jc)=0;
F=Q*x+f;
w=-F-lamb*sign(x);
J=find(x~=0);
if ~isempty(Jc)
    w(Jc)=(F(Jc))/max(abs(F(Jc)))*(lamb)*0.5-F(Jc);
end
r=-w;
tj=0;
count=1;
trecord=0;
while(tj<tmax)
    count        = count+1;    

    QJJ=Q(J,J);
    signxJ       = sign(-QJJ*x(J)-f(J)-w(J)-tj*r(J));
    if ~isempty(J)
        uv=QJJ\([-w(J)-f(J)-lamb*signxJ,-r(J)]);
        u=uv(:,1); v=uv(:,2);   %x=u+vt
        QJ=Q(:,J);
        a=QJ*u+f+w;
        b=QJ*v+r;
    else
        u=[];v=[];
        a=w+f;
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

