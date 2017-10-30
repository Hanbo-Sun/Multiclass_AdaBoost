function [Label, Err] = predAdaBoost(abClassifier, X, Y)
N = size(X, 1);

if nargin < 3
    Y = [];
end

p = length(unique(Y));
M = abClassifier.nWC;
Label = zeros(N, p);
%LabM = zeros(N, M);

for i=1:M
    tid=predict(abClassifier.WeakClas{i},X);
    for j = 1:N
        Label(j,tid(j))=Label(j,tid(j))+abClassifier.Weight(i);
    end
    %disp(Label)
end

[v,Label]=max(Label,[],2);
%disp(sum(Label))
Err=sum(Label~=Y)/N;



%for i = 1:M
 %   LabM(:,i) = abClassifier.Weight(i)*predict(abClassifier.WeakClas{i},X);
%end



%LabM = sum(LabM, 2);
%idx = logical(LabM > 0);
%Label(idx) = 1;
%Label(~idx) = -1;

% 
%if ~isempty(Y)
 %   Err = logical(Label ~= Y);
   % Err = sum(Err)/N;
%end
%end
