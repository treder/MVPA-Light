function [clabel,dval] = test_libsvm(cf,X)
% Applies a LIBSVM classifier to test data and produces class labels and decision values.
% 
% Usage:
% [labels,dval] = test_libsvm(cf,X)
% 
%Parameters:
% cf             - classifier. See train_libsvm
% X              - [number of samples x number of features] matrix of
%                  test samples
%Output:
% clabel        - predicted class labels
% dval          - decision values

[clabel, ~, dval] = svmpredict(zeros(size(X,1),1), X, cf,'-q');

% Note that dvals might be sign-reversed in some cases,
% see http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f430
% To fix this behavior, we inspect cf.Labels: Label(1) denotes the positive 
% class (should be 1)
if cf.Label(1) ~= 1
    % 1 is negative class, hence we need to flip dvals
    dval = -dval;
end