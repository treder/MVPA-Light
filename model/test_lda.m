function [clabel,dval,prob] = test_lda(cf,X)
% Applies an LDA classifier to test data and produces class labels,
% decision values, and posterior probabilities.
% 
% Usage:
% [clabel,dval] = test_lda(cf,X)
% 
%Parameters:
% cf             - struct describing the classifier obtained from training 
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

if cf.prob==1 && nargout>2
    % To obtain posterior probabilities, we evaluate a multivariate normal
    % pdf at the test data point. As decision values, we output relative
    % class probabilities for class 1
    prob1= mvnpdf(X, cf.mu1', cf.Sw/cf.n);
    prob2= mvnpdf(X, cf.mu2', cf.Sw/cf.n);
    prob= prob1*cf.prior1 ./ (prob1*cf.prior1 + prob2*cf.prior2);

end