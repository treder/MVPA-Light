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

[clabel, ~, dval] = predict(ones(size(X,1),1), sparse(X), cf.model, '-q');

% LIBLINEAR outputs 0 and 1, need to recode as 1 and 2
clabel(clabel==0) = 2;

% Note that dvals might be sign-reversed in some cases,
% see http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f430
% and https://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html
% To fix this behavior, we inspect cf.Labels: Label(1) denotes the positive 
% class (should be 1)
if cf.model.Label(1) ~= 1
    % 1 is negative class, hence we need to flip dvals
    dval = -dval;
end