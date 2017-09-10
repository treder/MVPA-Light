function [clabel,dval,post] = lda_classifier(Xtrain,clabel,Xtest,gamma)
% Linear discriminant analysis. Convenience function that does training and
% testing in one go.
% 
% Usage:
% [label,dval,post] = lda_classifier(train,labels,test,gamma)
% 
%Parameters:
% Xtrain         - [samples x features] matrix of training samples
% clabel         - [number of samples] vector of class labels containing 
%                  1's (class 1) and 2's (class 2)
% Xtest          - [number of samples x number of features] matrix of
%                  test samples
%Optional parameters:
% gamma           - regularisation parameter between 0 and 1 (where 0 means
%                   no regularisation and 1 means full max regularisation).
%                   If 'auto' then the regularisation parameter is
%                   calculated automatically using the Ledoit-Wolf formula(
%                   function cov1para.m)
%
%Output:
% clabel        - predicted class labels (1's and 2's)
% dval          - decision values, i.e. distances to the hyperplane
% post          - posterior class probabilities


% (c) Matthias Treder 2017

if ~exist('Xtest','var') || isempty(Xtest)
    Xtest = [];
end

if ~exist('gamma','var') || isempty(gamma)
    gamma = 0;
end

% Train LDA
cf = train_lda(struct('gamma',gamma),Xtrain,clabel);

% Test LDA
if nargout < 3
    [clabel,dval] = test_lda(cf,Xtest);
else
    [clabel,dval,post] = test_lda(cf,Xtest);
end
