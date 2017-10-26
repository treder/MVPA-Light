function [clabel,dval] = test_logreg(cf,X)
% Applies an Logistic Regression classifier to test data and produces class labels,
% decision values.
% 
% Usage:
% [labels,dval] = test_logreg(cf,X)
% 
%Parameters:
% cfy            - struct describing the classifier obtained from training 
%                  data. Must contain at least the fields w and b, 
%                  see train_lda
% X              - [number of samples x number of features] matrix of
%                  test samples
%
%Output:
% clabel        - predicted class labels (1's and 2's)
% dval          - decision values, i.e. distances to the hyperplane

if cf.zscore
    X = bsxfun(@minus, X, cf.mean);
    X = bsxfun(@rdivide, X, cf.std);
end

dval = X*cf.w + cf.b; % unlike LDA, b needs to be added here
clabel= double(dval >= 0) + 2*double(dval < 0);

