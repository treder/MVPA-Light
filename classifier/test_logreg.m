function [clabel,dval,prob] = test_logreg(cf,X)
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
% prob          - class probabilities

dval = X*cf.w + cf.b;
clabel= double(dval >= 0) + 2*double(dval < 0);

if nargout>2
    prob = 0.5 + 0.5 * tanh(0.5 * dval);
end

