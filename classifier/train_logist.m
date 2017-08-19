function cf = train_logist(X,labels,param)
% Trains a logistic regression classifier. This is a wrapper for Lucas
% Parra's logist.m function.
%
% Usage:
% cf = train_logist(X,labels,<param>)
% 
%Parameters:
% X              - [number of samples x number of features] matrix of
%                  training samples
% labels         - [number of samples] vector of class labels containing 
%                  1's (class 1) and -1's (class 2)
%
% param          - struct with hyperparameters (see logist.m for
%                  description)
% .lambda        - regularisation parameter
% .eigvalratio   - cut-off ratio of highest-to-lowest eigenvalue
%
%Output:
% cf - struct specifying the classifier with the following fields:
% classifier   - 'lda', type of the classifier
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold 

[v,~] = logist(X, labels(:)==1, [], 0, param.lambda, param.eigvalratio);

cf= struct();
cf.classifier= 'Logistic Regression';
cf.w= v(1:end-1);
cf.b= v(end);
