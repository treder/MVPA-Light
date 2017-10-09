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
%
%Output:
% clabel        - predicted class labels
% dval          - decision values

[clabel, ~, dval] = svmpredict(zeros(size(X,1),1), X, cf,'-q');

% Flip sign such that class 1 > 0 and class 2 < 0
dval = -dval;