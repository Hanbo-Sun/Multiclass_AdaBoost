function abClassifier = buildAdaBoost(trnX, trnY, iter, tstX, tstY)
if nargin < 4 % # of parameters
    tstX = [];
    tstY = [];
end
abClassifier = initAdaBoost(iter);
K=length(unique([trnY;tstY]));
N = size(trnX, 1); % Number of training samples
sampleWeight = repmat(1/N, N, 1);

for i = 1:iter
    if (mod(i,30)==0)
        disp(i);
    end
    weakClassifier = fitctree(trnX, trnY,'weights', sampleWeight);
    abClassifier.WeakClas{i} = weakClassifier;
    abClassifier.nWC = i;
    % Compute the weight of this classifier
    [t,temp_error]=predAdaBoost(abClassifier,trnX,trnY);
    %temp_error=sum((temp_pre~=trnY).*(sampleWeight))/sum(sampleWeight);
    %disp(temp_error)
    ide=(t==trnY);
    idne=~ide;
    if (temp_error==0)
        temp_error = temp_error+0.01;
    end
    %disp(idne);
    
    %temp_error=kfoldLoss(weakClassifier);
    
    abClassifier.Weight(i) = log((1-temp_error)/temp_error)+log(K-1);
    %disp(abClassifier.Weight(i))
    % Update sample weight
    temeq=exp((-(K-1)/K) * abClassifier.Weight(i));
    temneq = exp((1/K) * abClassifier.Weight(i));
    sampleWeight(ide)=sampleWeight(ide)*temeq;
    sampleWeight(idne)=sampleWeight(idne)*temneq;
    sampleWeight = sampleWeight./sum(sampleWeight); % Normalized

    %disp(sampleWeight);
    
    %label = predict(weakClassifier,trnX);
    %tmpSampleWeight = -1*abClassifier.Weight(i)*(trnY.*label); % N x 1
    %tmpSampleWeight = sampleWeight.*exp(tmpSampleWeight); % N x 1
    %sampleWeight = tmpSampleWeight./sum(tmpSampleWeight); % Normalized
    
    % Predict on training data
    [ttt, abClassifier.trnErr(i)] = predAdaBoost(abClassifier, trnX, trnY);
    % Predict on test data
    if ~isempty(tstY)
        abClassifier.hasTestData = true;
        [ttt, abClassifier.tstErr(i)] = predAdaBoost(abClassifier, tstX, tstY);
    end
    % fprintf('\tIteration %d, Training error %f\n', i, abClassifier.trnErr(i));
end
end
