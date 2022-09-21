%
%
% dataId =0;
% switch dataId
%     case 1,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\covtype.libsvm.binary.mat')
%     case 2,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\avazu-app.mat')
%     case 3,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\epsilon_normalized.t.mat')
%     case 4,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\HIGGS.mat')
%     case 5,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\kdda.mat')
%     case 6,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\kddb.mat')
%     case 7,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\news20.binary.mat')
%     case 8,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\rcv1_test.binary.mat')
%     case 9,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\SUSY.mat')
%     case 10,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\url_combined.mat')
%     case 11,
%         train_data = load('F:\users\wang\matlab\optimization\ML\SVM\data\2_class\webspam.mat')
%     case 0,
%
% end
% % %%
% % %
% %
% A=train_data.data{1}(:,1:end-17);Y=train_data.data{2}(1:size(A,2));
% sample_id=zlength(train_data.data{2})-1;y=full(train_data.data{1}(:,sample_id));%+1*randn(size(A,1),1);
% %%
%

% 
% para1.countmax=2000;
% para1.itermax=10000;
% para1.lamb=1e-2*max(abs(A'*y));
% para1.delta=1e-3;   %A’A的条件数越大，它越大
% 
% para1.apgtol=2e-5;
% para1.tol=1e-5;
% 
% para1.kkt_err=1e-7;
% para1.CK=600;para1.CKmax=2000;
% para1.funtol=1e-12;
function x=pbc_l1hom(A,y,x0,para1)
%%
 
kkt_err=para1.kkt_err;
[m,n]=size(A);
CK=para1.CK;CKmax=para1.CKmax;
% f=@(x) (1/2*norm(A*x-y)^2+para1.lamb*sum(abs(x)));
% x0=zeros(n,1);
x=x0;

 
Ax=A*x;
kt=@(x) (A'*(A*x-y));kt1=@(x)(abs(kt(x))-para1.lamb);kkt=@(x) (x~=0).*(kt(x)+para1.lamb*sign(x))+kt1(x).*(kt1(x)>0).*(x==0);
 
fvalue=(1/2*norm(Ax-y)^2+para1.lamb*sum(abs(x)));
Fc=ones(n,1);
res=A'*(Ax-y);
Ay=A'*y;
reswhole=res+para1.lamb*sign(x);
W1=find(x==0&abs(res)>(para1.lamb));    W2=find(x~=0&abs(reswhole)>0);W3=[];
jiange=max(1,(round(log(n)/log(10))-2));
Fvalue=1/2*norm(y)^2;
Time=0;
iter=2;
while (iter<para1.itermax)
    W=[];
    if mod(iter,10)==9
        if sum(x~=0)>CK
            CK=min(CKmax,floor(1.1*CK));
        end
    end
    resW1=res(W1); resW2=reswhole(W2);
    [~,W1sort]=sort(abs(resW1));[~,W2sort]=sort(abs(resW2));
    
    W=union(W,W1(W1sort(end-min(length(W1),CK)+1:end)));
    W=union(W,W2(W2sort(end-min(length(W2),CK)+1:end)));
    Fc(W)=Fc(W)+1;
    lenW=length(W);
    AW=A(:,W);
    AWW=full(AW'*AW);
    if mod(iter,10)==1
        para1.delta=max_svdnum(AWW,1)/10^7;
    end
    %% homotopy跟踪前加proximal项
    %     ys=-(Ay(W)-AW'*Ax+AWW*x(W));
    %     u=lasso_apg_hom1(AWW,ys,x(W),para);
    %     value_drese=1/2*u'*(AWW*u)+ys'*u+para.lamb*sum(abs(u))-(1/2*x(W)'*(AWW*x(W))+ys'*x(W)+para.lamb*sum(abs(x(W))));
    %% FISTA前加proximal项
    ys=-(Ay(W)-AW'*Ax+AWW*x(W))-para1.delta*x(W);
    Q=AWW+para1.delta*speye(length(W));
    u=lasso_apg_hom2(Q,ys,x(W),para1);
    value_drese=1/2*u'*(Q*u)+ys'*u+para1.lamb*sum(abs(u))-(1/2*x(W)'*(Q*x(W))+ys'*x(W)+para1.lamb*sum(abs(x(W))))-1/2*para1.delta*norm(x(W)-u)^2;
    %%
    fvalue=fvalue+value_drese;
    
    if  value_drese>0
        fprintf('subproblem solved low-preccionly')
    end
    %         fvalue-fs
    
    Axadd=AW*(u-x(W));
    Ax=Ax+Axadd;
    
    x(W)=u;
    
    if iter>20&&mod(iter,jiange)~=0
        ischoose=mod(iter,jiange);
        if ischoose==1;
            W3=setdiff(union(W1,W2),W);
            AW3=A(:,W3);
        end
        if ~isempty(W3)
            res(W3)=AW3'*(Ax-y);
            reswhole(W3)=res(W3)+para1.lamb*sign(x(W3));
        end
        res(W)=AW'*Ax-Ay(W);
        reswhole(W)=res(W)+para1.lamb*sign(x(W));
    else
        res=A'*(Ax-y);
        reswhole=res+para1.lamb*sign(x);
        W1=find(x==0&abs(res)>(para1.lamb));    W2=find(x~=0&abs(reswhole)>0);
        
        kkt_cond=sqrt(norm(reswhole(W2))^2+norm(abs(res(W1))-para1.lamb)^2)
        if kkt_cond<kkt_err
            break
        end
    end
    Fvalue(end+1)=fvalue;
     
    if abs(value_drese/(1+abs(fvalue)))<para1.funtol&(mod(iter-1,jiange)==0||iter<20)
        break;
    end
    iter=iter+1;
end





