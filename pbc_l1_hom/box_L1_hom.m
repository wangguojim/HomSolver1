function x=box_L1_hom(Q,r,xinit,ld,ud,para)
%该函数通过参数积极集方法求解
%APG-Homotopy算法求解Lasso问题
%min 1/2||Ax-y||^2+lambda*||x||_1  s.t. ld<=x<=ud
%% parameters initilization
lambda=para.lambda;
x=xinit;
n=length(xinit);
epsilon=10^-10;   %设置为10^-8效果较好
tmax=1;
%%
truntol=norm(x)*1e-5;
Jl=find(x<=(ld+truntol));Ju=find(x>=(ud-truntol));
J=find(x>(ld+truntol)&x<(ud-truntol)&x~=0);J0=find(x>(ld+truntol)&x<(ud-truntol)&x==0);
x(Jl)=ld(Jl);x(Ju)=ud(Ju);x(J0)=0;
res=Q*x+r;F1=(ld==0);F2=(ud==0);
w=-res-lambda*sign(x);
if ~isempty(Jl);    w(Jl)=w(Jl)-lambda*F1(Jl)+10;end
if ~isempty(Ju);   w(Ju)=w(Ju)+lambda*F1(Ju)-10;end
if ~isempty(J0);   w(J0)=(res(J0))/max(abs(res(J0)))*(lambda*0.5)-res(J0);end
aw=-w;
tj=0;
count=1;
trecord=0;

while(tj<tmax)
    count        = count+1;
    Qx=Q*x;
    QJJ=Q(J,J);QJ=Q(:,J);
    signxJ       = sign(-Qx(J)-r(J)-w(J)-tj*aw(J));
    if ~isempty(J)
        QxJc=Qx(J)-QJJ*x(J);
        uv=QJJ\([-QxJc-r(J)-w(J)-lambda*signxJ,-aw(J)]);
        u=uv(:,1); v=uv(:,2);   %x=u+vt
        a=Qx-tj*(QJ*v)+r+w;
        b=QJ*v+aw;
    else
        u=[];v=[];
        a=Qx+r+w;
        b=aw;
    end
    % find the additional index
    aJl=a(Jl)+lambda*(sign(ld(Jl))+F1(Jl));bJl=b(Jl);
    aJu=a(Ju)+lambda*(sign(ud(Ju))-F2(Ju));bJu=b(Ju);
    bJ0=b(J0);aJ0=a(J0)-lambda*sign(bJ0);  %res=a+tb
    %% -------------------------------------------------------%%
    if ~isempty(J)
    tallx1=-u./v;
    tupx1=find(tallx1>trecord(end)+10*epsilon);
    [tjx1,idx1]=min(tallx1(tupx1)) ;
    idx1=tupx1(idx1);
    
    tallx2=(ld(J)-u)./v;
    tupx2=find(tallx2>trecord(end)+10*epsilon);
    [tjx2,idx2]=min(tallx2(tupx2)) ;
    idx2=tupx2(idx2);
    
    tallx3=(ud(J)-u)./v;
    tupx3=find(tallx3>trecord(end)+10*epsilon);
    [tjx3,idx3]=min(tallx3(tupx3)) ;
    idx3=tupx3(idx3);
    else
        tjx1=106;
        tjx2=107;
        tjx3=108;
    end        
    %--------------------------------------------%
    tresJ0=(-aJ0)./bJ0;
    tupresJ0=find(tresJ0>trecord(end)+10*epsilon);
    [tjresJ0,idresJ0]=min(tresJ0(tupresJ0)) ;
    idresJ0=tupresJ0(idresJ0);
    
    tresJl=-aJl./bJl;
    tupresJl=find(tresJl>trecord(end)+10*epsilon);
    [tjresJl,idresJl]=min(tresJl(tupresJl)) ;
    idresJl=tupresJl(idresJl);
    
    tresJu=-aJu./bJu;
    tupresJu=find(tresJu>trecord(end)+10*epsilon);
    [tjresJu,idresJu]=min(tresJu(tupresJu)) ;
    idresJu=tupresJu(idresJu);
    %-------------------------------------------%
    if isempty(tjx1); tjx1=100; end
    if isempty(tjx2); tjx2=101; end
    if isempty(tjx3); tjx3=102; end
     if isempty(tjresJ0); tjresJ0=103; end
    if isempty(tjresJl); tjresJl=104; end
    if isempty(tjresJu); tjresJu=105; end
   
    [tj,id]=min([tjx1,tjx2,tjx3,tjresJ0,tjresJl,tjresJu]);
    %% ---------------flash the active set-------------%    
    if(tj>=1)
        tj=1;
        x(J)          = u+tj*v ;
    else
        x(J)          = u+tj*v ;
        if (id==1)
            Jout=J(idx1);
            x(Jout)=0;
            J=setdiff(J,Jout);
            if (ld(Jout)==0)
                Jl(end+1)=Jout;
            elseif  (ud(Jout)==0)
                Ju(end+1)=Jout;
            else
                J0(end+1)=Jout;
            end
        elseif (id==2)
            Jout=J(idx2);
            J=setdiff(J,Jout);
            x(Jout)=ld(Jout);
            Jl(end+1)=Jout;
        elseif (id==3)
            Jout=J(idx3);
            J=setdiff(J,Jout);
            x(Jout)=ud(Jout);
            Ju(end+1)=Jout;
        elseif (id==4)
            J0out=J0(idresJ0);
            J0=setdiff(J0,J0out);
            J(end+1)=J0out;
        elseif (id==5)
            Jlout=Jl(idresJl);
            Jl=setdiff(Jl,Jlout);
            J(end+1)=Jlout;            
        elseif (id==6)
            Juout=Ju(idresJu);
            Ju=setdiff(Ju,Juout);
            J(end+1)=Juout;            
        end
    end
    trecord(count) = tj;
end

max(res(Ju))
max(abs(res(J0)))