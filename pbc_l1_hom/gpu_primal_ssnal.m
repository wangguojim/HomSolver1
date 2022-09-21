function xk=gpu_primal_ssnal(A,y,para,xinit)
% para.counMAX=2000;
% para.tol=1e-10;
%%
xk=xinit;
n=length(xinit);
count=0;
err=1;
while (err>para.tol&&count<para.counMAX)
    
    gk=A'*(A*xk-y)+ para.delta*(xk-xinit);
    axk=abs(xk);
    hk0=gk +para.lamb*sign(xk );
    hk=abs(hk0);
    agk=abs(gk )-para.lamb;
    Ja =find(axk<=hk & axk>=agk );
    Jb =find(axk<=hk & axk<agk );
    Jc =find(axk>hk & hk>=agk );
    Jd =find(axk>hk & hk<agk );
    fai_xk=zeros(n,1);
    fai_xk (Ja )= xk(Ja);
    Jbd=union(Jb,Jd);
    fai_xk( Jbd )=gk(Jbd)-para.lamb*sign(gk(Jbd));
    fai_xk( Jc )=hk0(Jc);
    W=union(Jbd,Jc);
    AW=A(:,W);
    gpuAW=gpuArray(AW);
    gpuHW=gpuAW'*gpuAW+para.delta* gpuArray(eye(length(W)));
    HW=AW'*AW+para.delta* eye(length(W));
    
    NJ=Ja(xk(Ja)~=0);
    if isempty(NJ)
        fk1=fai_xk(W);
    else
        fk1=fai_xk(W)-AW'*(A(:,NJ)*xk(NJ));
    end
    gpuHW\gpuArray(fk1);
    s11=HW\fk1;
    pk=zeros(n,1);
    pk(Ja)=-xk(Ja);
    pk(W)=-s11;
    min1=max(max(min(abs(xk),abs(gk+para.lamb*sign(xk))),abs(gk)-para.lamb));
    ID=union(NJ,W);
    Hpk=A'*(A(:,ID)*pk(ID))+para.delta*pk;
    bk=1;
    xk1=xk+bk*pk;
    axk1=abs(xk1);
    hk1=abs(gk+bk*Hpk +para.lamb*sign(xk1 ));
    agk1=abs(gk+bk*Hpk)-para.lamb;
    min2=max(max(min(axk1,hk1),agk1));
    while min1<min2
        bk=bk/2;
        xk1=xk+bk*pk;
        axk1=abs(xk1);
        hk1=abs(gk+bk*Hpk +para.lamb*sign(xk1 ));
        agk1=abs(gk+bk*Hpk)-para.lamb;
        min2=max(max(min(axk1,hk1),agk1));
    end
    xk=xk1;
    err=min2;
    count=count+1;
end