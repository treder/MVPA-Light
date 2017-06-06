function [label,dval,post] = lda_classifier(Xtrain,labels,Xtest,gamma)
% Linear discriminant analysis. Convenience function that does training and
% testing in one go.
% 
% Usage:
% [label,dval,post] = lda_classifier(train,labels,test,gamma)
% 
%Parameters:
% Xtrain         - [number of samples x number of features] matrix of
%                  training samples
% labels         - [number of samples] vector of class labels containing 
%                  1's (class 1) and -1's (class 2)
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
% label         - predicted class labels (1's and -1's)
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
cf = train_lda(struct('gamma',gamma),Xtrain,labels);

% Test LDA
if nargout < 3
    [label,dval] = test_lda(cf,Xtest);
else
    [label,dval,post] = test_lda(cf,Xtest);
end
