function [label,dval] = test_lda(cf,X)
% Applies an LDA classifier to test data and produces class labels,
% decision values, and posterior probabilities.
% 
% Usage:
% [labels,dval,post] = test_lda(cf,X)
% 
%Parameters:
% cf             - struct describing the classifier obtained from training 
%                  data. Must contain at least the fields w and b, 
%                  see train_lda
% X              - [number of samples x number of features] matrix of
%                  test samples
%
%Output:
% label         - predicted class labels (1's and 2's)
% dval          - decision values, i.e. distances to the hyperplane or
%                 class probabilities

dval = X*cf.w - cf.b;
label= double(dval >= 0) + 2*double(dval < 0);

if cf.prob==1
    % To obtain posterior probabilities, we evaluate a multivariate normal
    % pdf at the test data point. As decision values, we output relative
    % class probabilities for class 1
    prob1= mvnpdf(X, cf.mu1', cf.C);
    prob2= mvnpdf(X, cf.mu2', cf.C);
    dval= prob1*cf.prior1 ./ (prob1*cf.prior1 + prob2*cf.prior2);

    % The following code does not work, since normpdf on the projections
    % does not yield results that are equivalent to mvnpdf on the
    % non-projections
    % For the posterior probability, we do not need to evaluate the
    % multivariate normal pdf. Instead, we can simply evaluate the 1D
    % normal pdf after projecting the class means and C onto w - this is
    % faster
%     prob1= normpdf(X * cf.w, cf.m1, cf.sigma);
%     prob2= normpdf(X * cf.w, cf.m2, cf.sigma);
%     dval= prob1*cf.prior1 ./ (prob1*cf.prior1 + prob2*cf.prior2);

end