function [clabel,dval] = test_linear_svm(cf,X)
% Applies a SVM to test data and produces class labels and decision values.
% 
% Usage:
% [labels,dval,post] = test_svm(cf,X)
% 
%Parameters:
% cf             - classifier. See train_svm
% X              - [number of samples x number of features] matrix of
%                  test samples
%
%Output:
% clabel        - predicted class labels (1's and 2's)
% dval          - decision values, i.e. distances to the hyperplane


dval = X*cf.w + cf.b; % unlike LDA, b needs to be added here
clabel= double(dval >= 0) + 2*double(dval < 0);
