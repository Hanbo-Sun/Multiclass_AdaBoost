clear all

z = dlmread('spambase.data',',');
rng(0); % initialize the random number generator
rp = randperm(size(z,1)); % random permutation of indices
z = z(rp,:); % shuffle the rows of z
x = z(:,1:end-1);
y = z(:,end);

[n,m]=size(x);
n_train=2000;
n_test=n-n_train;
k=2;
j=m;

%quantize values to 1 or 2
x_med=median(x);
for i =1:n
    less=find(x(i,:)<=x_med);
    large=find(x(i,:)>x_med);
    x(i,less)=1;
    x(i,large)=2;
end

%partition of x to train and test set
x_train=x(1:n_train,:);
y_train=y(1:n_train,:);
x_test=x(n_train+1:n,:);
y_test=y(n_train+1:n,:);

%get the frequency of every class in training set
pk=[0;0];
nk=[sum(y_train==0);sum(y_train==1)];
pk(1)=nk(1)/n_train;
pk(2)=nk(2)/n_train;

%get condition probability of ?2?
gk=zeros(k,j);
for i=1:k
    for ii=1:j
        gk(i,ii)=sum(x_train(y_train==i-1,ii)==2);
        gk(i,ii)=(gk(i,ii)+1)/(nk(i)+2);
    end
end

%get predict value
y_pred=zeros(n_test,1);
for i=1:n_test
    f=[log(pk(1));log(pk(2))]; 
    for ii=1:m
        for iii=1:k
            aa=log((gk(iii,ii)^(x_test(i,ii)==2))); 
            bb=log((1-gk(iii,ii))^(x_test(i,ii)==1));
            f(iii)=f(iii)+aa+bb; % if x(i,ii)== 2 then f(k)+log(g), else f(k)+log(1-g)
        end
    end
    if f(1)<f(2)
        y_pred(i)=1; 
    end
end

accuracy=sum(y_pred==y_test)/n_test;
%test error
error_test=1-accuracy;
% always predict the majority class
check_error=min(sum(y_test)/n_test,1-sum(y_test)/n_test);
