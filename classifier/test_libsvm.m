function [clabel,dval] = test_libsvm(cf,X)
% Applies a LIBSVM classifier to test data and produces class labels and decision values.
% 
% Usage:
% [labels,dval] = test_libsvm(cf,X)
% 
%Parameters:
% cf             - classifier. See train_libsvm
% X              - [samples x features] data matrix  -OR-
%                  [train samples x test samples] kernel matrix
%Output:
% clabel        - predicted class labels
% dval          - decision values

if cf.kernel_type == 4
    % kernel has been precomputed - we only pass on the kernel matrix, not
    % the data
    error('dealing with precomputed kernel matrix needs some attention here')
    nK = size(cf.kernel_matrix,1);
    [clabel, ~, dval] = svmpredict(zeros(nK,1), [(1:nK)', X], cf.model,'-q');
else
    % kernel is compute in svmtrain, pass on data
    [clabel, ~, dval] = svmpredict(zeros(size(X,1),1), X, cf.model,'-q');
end

% Note that dvals might be sign-reversed in some cases,
% see http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f430
% To fix this behavior, we inspect cf.Labels: Label(1) denotes the positive 
% class (should be 1)
if cf.model.Label(1) ~= 1
    % 1 is negative class, hence we need to flip dvals
    dval = -dval;
end