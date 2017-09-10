function [labels,dval] = test_svm(cf,X)
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

par= ['-q']; % no output

[labels,~,dval] = svmpredict(zeros(size(X,1),1),X,cf.model,par);

%% FIX TODO ENSURE class labels are 1 and 2 (not 1 and -1) !!!
