 
function x=lasso_gcd(A,y,para,xinit)
x=xinit;
grad=-(A'*y);
for outiter=1:para.IterMax
    W1=find(x~=0);
    W2=find(abs(grad)>0.999*para.lamb);
    W=union(W1,W2);
    AW=A(:,W);
    gradW=grad(W);
    xW=x(W);
    KKT_err=zeros(length(W),1);
    for iter =1:min(para.Initer,2*length(W))
        
        J1=(xW~=0);
        J2=xW==0;
        KKT_err(J1)= abs(gradW(J1) +para.lamb*sign( xW(J1)) );
        KKT_err(J2)= abs(gradW(J2))-para.lamb  ;
        [kkt_err1,idx]=max(KKT_err);
%         if kkt_err1<para.KKT_tol
%             break
%         end
        
        Ai=AW(:,idx);
        Li=Ai'*Ai;
        ai=xW(idx)-1/Li*gradW(idx);
        bi=(ai-sign(ai)*para.lamb/Li)*(abs(ai)>(para.lamb/Li));
        xiold=xW(idx);
        
        if bi*xW(idx)<0
            xW(idx)=0;
        else
            xW(idx)=bi;
        end
        
        gradW=gradW+AW'*(Ai*( xW(idx)-xiold));
            
    end
   
    x(W)=xW;
    grad =A'*(AW* x(W)-y)   ;
 
     if mod(outiter,2)==0
         W1=W(xW~=0);
         kkt_whole= norm(grad(W1)+para.lamb*sign(x(W1)))+norm(max(0,  abs(grad)-para.lamb )) 
         if kkt_whole<para.KKT_tol
             break
         end
     end
end
  
