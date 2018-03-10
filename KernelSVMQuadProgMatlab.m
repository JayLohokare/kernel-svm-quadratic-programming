load q3_1_data.mat
C = 0.1;
[w,b,a,obj] = SVM(trD,trLb,C);
val = valD'*w+b;

truePos=sum((val>=0)&(valLb>0));
falseNeg=sum((val<0)&(valLb>0));
trueNeg=sum((val<0)&(valLb<0));
falsePos=sum((val>=0)&(valLb<0));

accuracy=(truePos+trueNeg)/length(valLb);

fprintf('Value of C %f:\n',C);
fprintf('The model accuracy is %.2f\n', accuracy*100);
fprintf('Objective value is %.2f\n', obj);
fprintf('The number of SVs is %.2f\n', sum(a>1e-1));

fprintf('Confusion matrix is \n');
fprintf('%d %d\n%d %d\n',truePos,falsePos,falseNeg,trueNeg);


C = 10;
[w,b,a,obj] = SVM(trD,trLb,C);
val = valD'*w+b;

truePos=sum((val>=0)&(valLb>0));
falseNeg=sum((val<0)&(valLb>0));
trueNeg=sum((val<0)&(valLb<0));
falsePos=sum((val>=0)&(valLb<0));

accuracy=(truePos+trueNeg)/length(valLb);

fprintf('Value of C %f:\n',C);
fprintf('The model accuracy is %.2f\n', accuracy*100);
fprintf('Objective value is %.2f\n', obj);
fprintf('The number of SVs is %.2f\n', sum(a>1e-1));

fprintf('Confusion matrix is \n');
fprintf('%d %d\n%d %d\n',truePos,falsePos,falseNeg,trueNeg);
function [w,b,a,obj]=SVM(X,Y,C)
K=X'*X;
options.Display='off';
[a,obj,flag]=quadprog(diag(Y)*K*diag(Y),-ones(size(Y,1),1),[],[],Y',0,zeros(size(Y,1),1),C*ones(size(Y,1),1),[],options);
obj=-obj;
[~,sv]=max(min(a,C-a));
b=Y(sv)-K(sv,:)*diag(Y)*a;
w=X*diag(Y)*a;
end
