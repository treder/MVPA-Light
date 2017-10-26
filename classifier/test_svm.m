function [predlabel,dval] = test_svm(cf,X)
% Applies a SVM to test data and produces class labels and decision values.
% 
% Usage:
% [predlabel,dval] = test_svm(cf,X)
% 
%Parameters:
% cf             - classifier. See train_svm
% X              - [samples x features] matrix of test data
%
%Output:
% predlabel     - predicted class labels (1's and 2's)
% dval          - decision values, i.e. distances to the hyperplane

if strcmp(cf.kernel,'linear')
    dval = X*cf.w + cf.b;
else
    dval = cf.alpha_y' * cf.kernelfun(cf, cf.support_vectors, X)  + cf.b;
end

predlabel= double(dval >= 0) + 2*double(dval < 0);