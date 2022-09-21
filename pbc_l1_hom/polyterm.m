function A=polyterm(m,n)
% n=3;m=9;
A=zeros(n+1,m);
A(:,1)=(0:n)';
for i=1:m-1
    B=[];
    for j=1:size(A,1)
        s=A(j,:);
        num=sum(s);
        H=ones(n-num+1,1)*s;
        H(:,i+1)=(0:n-num);
        B(end+1:end+size(H,1),:)=H;
    end
    A=B;
end
A=A(2:end,:);
