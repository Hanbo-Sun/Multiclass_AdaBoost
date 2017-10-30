% Weak classifiers: Multi-class NB
%just in case

% Hanbo Sun Dec/03/2016

clear all;
load fisheriris.mat
x=meas;
y=[];
J=size(x,2);
k=3; % # class
for i =1:150
    if length(species{i})==6
        y(i)=1;
    else
        if length(species{i})==10
            y(i)=2;
        else
            y(i)=3;
        end
    end
end
y=y';

%quantize values to 1 or 2
xm=median(x);
xm=repmat(xm,150,1);
x(x<=xm)=1;
le=(x<=xm);
la=(x>xm);
x(le)=1;
x(la)=2;

rp = randperm(size(x,1)); % random permutation of indices
x = x(rp,:); % shuffle the rows of z
y=y(rp,:);
%partition to train and test set
xtr=x(1:90,:);
ytr=y(1:90,:);
xte=x(91:end,:);
yte=y(91:end,:);
ntr=size(xtr,1);
nte=size(xte,1);

%get the frequency of every class in training set
nk=[sum(ytr==1);sum(ytr==2);sum(ytr==3)];
pk=[nk(1)/ntr,nk(2)/ntr,nk(3)/ntr];

%get condition probability of ?2?
gk=zeros(k,J);
for i=1:k
    for j=1:J
        gk(i,j)=sum(xtr(ytr==i,j)==2);
        gk(i,j)=(gk(i,j)+1)/(nk(i)+2);
    end
end

%get predict value

yp=zeros(nte,1);
for i=1:nte
    f=[log(pk(1));log(pk(2));log(pk(3))]; 
    for ii=1:J
        for kk=1:k
            aa=log((gk(kk,ii)^(xte(i,ii)==2))); 
            bb=log((1-gk(kk,ii))^(xte(i,ii)==1));
            f(kk)=f(kk)+aa+bb; % if x(i,ii)== 2 then f(k)+log(g), else f(k)+log(1-g)
        end
    end
    [v,id]=max(f);
    yp(i)=id; 
end

ac=sum(yp==yte)/nte;
%test error
err=1-ac; %0.26-0.35
% always predict the majority class
check_err=min([(sum(yte==1)+sum(yte==2))/nte,(sum(yte==1)+sum(yte==3))/nte...
    ,(sum(yte==2)+sum(yte==3))/nte]); %0.65 or so






