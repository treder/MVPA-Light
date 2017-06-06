function [label,dval,varargout] = test_lda(cf,X)
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
% label         - predicted class labels (1's and -1's)
% dval          - decision values, i.e. distances to the hyperplane
% post          - posterior class probabilities

dval = X*cf.w - cf.b;
label= sign(dval);

if nargout>2
    % Calculate posterior probabilities (probability for a sample to be
    % class 1)
    
    % The prior probabilities are calculated from the training
    % data using the proportion of samples in each class 
    prior1 = cf.N1/(cf.N1+cf.N2);
    prior2 = 1 - prior1;
    
    prob1= mvnpdf(X',cf.mu1,cf.C);
    prob2= mvnpdf(X',cf.mu2,cf.C);
    varargout{1}= prob1*prior1 / (prob1*prior1 + prob2*prior2);
end