function [label,dval] = test_logreg(cf,X)
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
% label         - predicted class labels (1's and -1's)
% dval          - decision values, i.e. distances to the hyperplane

dval = X*cf.w + cf.b; % unlike LDA, b needs to be added here
label= double(dval >= 0) + 2*double(dval < 0);

