load q3_1_data.mat

C = 10;

[w,b,a,obj] = SVM(trD,trLb,C);

val = valD'*w+b;






function [w,b,a,obj]=SVM(X,Y,C)
opts.Display='off';
[alpha,obj,flag]=quadprog(X'*X,-ones(size(Y,1),1),[],[],Y',0,zeros(size(Y,1),1),C*ones(size(Y,1),1),[],opts);
obj=-obj;
[~,freesvs]=max(min(alpha,C-alpha));
b=Y(freesvs)-K(freesvs,:)*diag(Y)*alpha;
%[a,b,obj]=KernelSVM2(K,Y,C);
w=X*diag(Y)*a;
end

