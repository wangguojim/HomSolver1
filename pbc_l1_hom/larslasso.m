function [x,X,Lambdarecord]=larslasso(A,y,cas,cas_num)
% this program can solve LASSOproblem with LARS HOMOTOPY algorithms
% the original problem is argmin_x f(x)=1/2*||y-Ax||^2+Lambda*||x||1
%input:
%    cas :has two choices
%       'Lambda'           :program will end until the Lambda is small than
%                           the inital Lambda
%       'nonzeronum'       :program will end until the number of nonzero
%                          elements of x is more than the numbeof nonzero
%                          elements of x we want to acchive.
%  cas_num 
%                          in the case 'Lambda',it is the value of Lambda.
%                          . in the case 'nonzeronum',it is the value of
%                          the number of nonzero elements of x.
%output:
%    X             :the value of all x to the Lambdarecord                   
%    x             :the value of x to the last Lambda
%    Lambdarecord  :all Lambda when cative set change
%
%
p=size(A,2);
eps=10^-9;
tof=false;
chcase={'Lambda','nonzeronum'};
if length(cas)==10
    Lambda=eps;
    nzn=cas_num;  
    tof=true;
elseif length(cas)==6
    Lambda=cas_num;
    nzn=p;    
else
    fprintf('please make sure the parameter is true')
    return;
end
[Lambdamax,id] = max(abs(A'*y));                  
Lambdarecord=Lambdamax;
x=zeros(p,1);
if Lambda>=Lambdamax
    X=x;
    Lambdarecord=Lambda;
    return;
end
J              = id;                                 %J is the active set
Jlength        = length(J);
count          = 1;

X(:,count)     = x;
nonzn=1;
while(nonzn<=nzn)
    count        = count+1;
    AJ           = A(:,J);
    CJ           = AJ'*AJ;
    signxJ       = sign(AJ'*(y-AJ*x(J)));
    [G,R,tru]    = cholesk(CJ);
    u            = solvep(G,R,AJ'*y);   %set u=[(AJ'*AJ)^-1]*AJ'*y  
    v            = solvep(G,R,signxJ);  %set v=[(AJ'*AJ)^-1]*signJ    
    Lambdaj      = 0;
    jinclude           = 0;
%% --------------------------------------------------------------%
% find the additional index 
    for j=1:p         
        if (sum([J-j]==0))==1
            continue;
        else            
            a      = A(:,j)'*(y-AJ*u); %set a=Aj'*(y-AJ*u)
            b      = A(:,j)'*AJ*v   ;  %set a=Aj'*AJ*v
            Re=Resru(a,b,1);
            if(Lambdaj<Re&&Re<Lambdarecord(end))
                Lambdaj = Re;
                jinclude      = j;
            end
        end
    end
    x(J)          = u-Lambdaj*v;
    
%% -------------------------------------------------------%%
    % find the deletion index 
    lzero=X(J,end).*x(J);
    L=length(lzero);
    idl=0;
    Lambdajc=Lambdaj;
    for l=1:L
        if lzero(l)<-eps
            lamb=Lambdarecord(end)-(Lambdarecord(end)-Lambdajc)*abs(X(J(l),end))/abs(X(J(l),end)-x(J(l)));            
            if lamb>Lambdaj
                Lambdaj=lamb;
                idl=l;
            end
        end
    end
    x(J)          = u-Lambdaj*v;
%% -----------------------------------------------------------%
% flash the active set
    if idl~=0
        nonzn=nonzn-1;    
        x(J(idl))=0;
        J=setdiff(J,J(idl));
    else
        nonzn=nonzn+1;
        if jinclude~=0
           J(end+1) = jinclude;
        end 
    end 
%% ----------------------------------------------------------%
     
    X(:,count)    = x;
    Lambdarecord(count) = Lambdaj;   
    if Lambdaj<eps&& Lambdaj>Lambda
        warning('Lambda is close to zero');
        break;
    elseif Lambdaj<Lambda && tof==false        
        x= ((Lambdarecord(end-1)-Lambda)*X(:,end)+...
            (Lambda-Lambdarecord(end))*X(:,end-1))/(Lambdarecord(end-1)-Lambdarecord(end));
        break;        
    end    
end
%% ========================================================== %%
% this function will solve the equation |c+b*Lambdaj|=C*Lambdaj for Lambdaj
function Lambdaj=Resru(a,b,C)
if (a>0)&&(b>0)&&(b<C)
    Lambdaj=a/(C-b);
elseif (a>0)&&(b<0)
    Lambdaj=a/(C-b);
elseif (a<0)&&(b<0)&&(b>-C)
    Lambdaj=-a/(C+b);
elseif (a<0)&&(b>0)
    Lambdaj=-a/(C+b);
else 
    Lambdaj=-inf;
    warning('this equqtion has no solution')
end
%% ===========================================================%%
% this function can solve linear equations system G*R*R'*G'*p=r,
% where G is a Permutation matrix,R is a low triangle matrix  
function p=solvep(G,R,r)
Rrow  = size(R,2);
r       = G'*r;
y        = r/R(1,1);         %solve Ry=G'*r
for i=2:Rrow              
    y(i)      = (r(i)-R(i,1:i-1)*y(1:i-1))/R(i,i);
end

u        = y/R(Rrow,Rrow);   %solve R'u=y
for i=2:Rrow
    u(Rrow-i+1) = (y(Rrow-i+1)-R(Rrow-i+2:Rrow,Rrow-i+1)'*u(Rrow-i+2:Rrow))/R(Rrow-i+1,Rrow-i+1);
end

p=G*u; 
%% ========================================================= %%
% this function will return the cholesky decompsition of a symmertic
% positive matrix A,ie. G'R'RG=A
function [G,R,tru]=cholesk(A)
tru=1;
Arow = size(A,2);
R           = eye(Arow);
G           = eye(Arow);
epss        = 10^-5;                             % make sure that A is suffecicently positive
for i=1:Arow  
   
    AA                 = A(i:end,i:end);    
    [~,index]          = max(diag(AA));    
    t                  = A(i+index-1,:);
    A(i+index-1,:)     = A(i,:);
    A(i,:)             = t;
    t                  = A(:,i+index-1);
    A(:,i+index-1)     = A(:,i);
    A(:,i)             = t;
    %A=GG*A*GG  
    K=A(i+1:end,i+1:end);
    if A(i,i)<epss
        tru=0;
        return ;        
    end
    a                  = sqrt(A(i,i));           %alpha   
    w                  = A(i+1:end,i)/a;
    RR(i+1:Arow,i)     = w;
    RR(i,i)            = a;
    A(i,i)             = 1;
    A(i,i+1:end)       = 0;
    A(i+1:end,i)       = 0;     
    A(i+1:end,i+1:end) = K-w*w' ;
    t                  = R(:,i+index-1);
    R(:,i+index-1)     = R(:,i);
    R(:,i)             = t;
    R(:,i)             = R*RR(:,i);
    %R=R*RR;  %Record Gi
    %G=G*GG; %Record Ri
    t                  = G(:,i+index-1);
    G(:,i+index-1)     = G(:,i);
    G(:,i)             = t;    
end
R(:,end)    = R(:,end)*sqrt(A(end,end));
R           = G'*R;