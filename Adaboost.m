% Multi-class Adaboost
% Ref: "Multi-class Adaboost" Zhu, J, etal(2009).
% Ref: https://github.com/jiyfeng/AdaBoost/blob/master/CV/buildCVMatrix.m
% Ref: function "fitctree"(MATLAB)
% Realization: Hanbo Sun Dec/03/2016 
clear all;
addpath 'AdaBoost-master/Boost';
addpath 'AdaBoost-master/Data';
addpath 'AdaBoost-master/CV';
addpath 'datasets';


%% Load data & preproess "Iris"
load fisheriris.mat;

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
rp = randperm(size(x,1)); % random permutation of indices
x = x(rp,:); % shuffle the rows of z
y=y(rp,:);
X=x;
Y=y;

%% Load data & preproess "Pendigits"
% load pendigits.mat
% X=[Xtr,Xte]';
% Y=[Ytr,Yte]';
% Y(Y==0) = 10;


%% Load data & preproess "Segmentation"
% A=dlmread('segmentation.txt');
% 
% rp = randperm(size(A,1)); % random permutation of indices
% A = A(rp,:); % shuffle the rows of z
% Y = A(:,1); 
% X=A(:,2:end);


%% Load data & preproess "Letter"
% A=dlmread('letter-recognition.txt');
% 
% rp = randperm(size(A,1)); % random permutation of indices
% A = A(rp,:); % shuffle the rows of z
% Y = A(:,1); 
% X=A(:,2:end);


%% main
%initialize
nfold = 10;
iter = 600;
tstError = zeros(nfold, iter);
trnError = zeros(nfold, iter);
[trnM, tstM] = buildCVMatrix(size(X, 1), nfold);

for n = 1:nfold
    fprintf('\tFold %d\n', n);
    idx_trn = logical(trnM(:, n) == 1);
    trnX = X(idx_trn, :);
    tstX = X(~idx_trn, :);
    trnY = Y(idx_trn);
    tstY = Y(~idx_trn);
    abClassifier = buildAdaBoost(trnX, trnY, iter, tstX, tstY);
    trnError(n, :) = abClassifier.trnErr;
    tstError(n, :) = abClassifier.tstErr;
end

plot(1:iter, mean(trnError, 1),'b');
hold on;
plot(1:iter, mean(tstError, 1),'r');
