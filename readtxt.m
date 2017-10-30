
bb=mean(tstError, 1);
%bb=abClassifier.Weight;
fid=fopen('SAMME-iris-err.txt','w');
fprintf(fid,'%f\n',bb);

