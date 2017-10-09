function [clabel,dval] = test_liblinear(cf,X)
% Applies a LIBLINEAR classifier to test data and produces class labels 
% and decision values.
% 
% Usage:
% [labels,dval] = test_liblinear(cf,X)
% 
%Parameters:
% cf             - classifier. See train_liblinear
% X              - [number of samples x number of features] matrix of
%                  test samples
%
%Output:
% clabel        - predicted class labels
% dval          - decision values

[clabel, ~, dval] = predict(zeros(size(X,1),1), sparse(X), cf, '-q');

% Flip sign such that class 1 > 0 and class 2 < 0
dval = -dval;
 