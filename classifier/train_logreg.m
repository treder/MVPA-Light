function [cf, b, stats] = train_logreg(cfg,X,label)
% Trains a logistic regression classifier with elastic net regularisation.
% The regularisation parameter lambda controls the strength of the
% regularisation. The parameter alpha is bounded between 0 and 1. It blends
% between blends between full L1/lasso regularisation (alpha=1) and
% L2/ridge regularisation (alpha close to 0). Must be strictly larger than
% 0. For intermediate values, both L1 and L2 regularisation applies. 
%
% Uses the MATLAB function lassoglm.
%
% Usage:
% cf = train_logreg(cfg,X,label)
% 
%Parameters:
% X              - [number of samples x number of features] matrix of
%                  training samples
% labels         - [number of samples] vector of class labels containing 
%                  1's (class 1) and -1's (class 2)
%
% param          - struct with hyperparameters:
% alpha          - blend between L1 regularisation (alpha=1) and L2
%                  regularisation (alpha approaches 0). 
%                  The closer to 1 alpha is set, the more sparse the 
%                  coefficient vector becomes. If sparsity is not required, 
%                  a small value of alpha is recommended since it speeds up
%                  training considerably (default 10^-10)
% nameval        - a cell array giving additional name-value pairs (e.g. 
%                  {'LambdaRatio' 100 'Link' 'logit'} passed to lassoglm.
%                  See the help of lassoglm for an explanation of the
%                  parameters
% K              - The hyperparameter lambda controlling the amount of 
%                  regularisation is found using a grid search based on 
%                  inner cross-validation. K sets the number of folds for 
%                  the cross-validation (default 5)
% numLambda      - the amount of lambda values that are checked during grid
%                  search

%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold 
%

[b, stats] = lassoglm(X, label(:)==1, 'binomial','alpha', cfg.alpha,...
    'CV',cfg.K, 'numLambda',cfg.numLambda, cfg.nameval{:}); 

% Select classifier weights according to the lambda yielding the lowest 
% deviance
cf.w= b(:,stats.IndexMinDeviance);
cf.b= stats.Intercept(stats.IndexMinDeviance);
cf.best_lambda= stats.LambdaMinDeviance;
