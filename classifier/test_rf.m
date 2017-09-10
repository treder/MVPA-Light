function [clabel,dval] = test_rf(cf,X)
% Applies a Random Forest to test data and produces class labels and decision values.
% 
% Usage:
% [labels,dval,post] = test_rf(cf,X)
% 
%Parameters:
% cf             - struct describing the classifier obtained from training 
%                  data. see train_rf
% X              - [number of samples x number of features] matrix of
%                  test samples
%
%Output:
% label         - predicted class labels (1's and -1's)
% dval          - decision values, i.e. distances to the hyperplane

clabel= cf.model.predict(X);

% Convert cell array of responses to double vector
clabel= arrayfun(@(x) str2double(x{1}), clabel);
dval = nan;
