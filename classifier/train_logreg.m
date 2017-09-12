function [cf, b, stats] = train_logreg(cfg,X,clabel)
% Trains a logistic regression classifier. To avoid overfitting, L2
% regularisation is used.
% To avoid overfitting, L1 or L2 regularisation can be used. L2
% regularisation penalises by the L2 norm of the weight vector w, or w'*w,
% whereas L1 regularisation penalises by its L1 norm, |w|.
% Unless a sparse w is needed, 
% Note that either 
%
% 

% In regularised logistic regression, the loss function can be defined as:
%      L(w,lambda) = SUM log(1+exp(-yi*w*xi)) + lambda * ||w||^2
%
% where w is the coefficient vector and lambda is the regularisation
% strength, and yi = {-1,+1} are the class labels.
%
% Usage:
% cf = train_logreg(cfg,X,clabel)
% 
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing 
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with hyperparameters:
% L2           - 
% L1           - lambda regularisation parameter using 


%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold 
%

% Reference:
% RE Fan, KW Chang, CJ Hsieh, XR Wang, CJ Lin (2008).
% LIBLINEAR: A library for large linear classification
% Journal of machine learning research 9 (Aug), 1871-1874

% Matthias Treder 2017

idx1 = (clabel == 1);
idx2 = (clabel == 2);

mv_setDefault(cfg,'normalise',1);
mv_setDefault(cfg,'intercept',1);
mv_setDefault(cfg,'CV',1);
% mv_setDefault(cfg,'lambda',2.^[-10:10]);
mv_setDefault(cfg,'lambda',1);

lambda = cfg.lambda;

if cfg.normalise
    X = zscore(X);
end

% We can reduce exp(-yi*w*xi) to exp(w*xi) by multiplying class 1 samples
% by -1
X(idx1,:) = -X(idx1,:);

% Augment X with intercept
if cfg.intercept
    X = cat(2,X, ones(size(X,1),1));
end

% Take vector connecting the class means as initial guess for speeding up
% convergence
w0 = mean(X(idx1,:)) -  mean(X(idx2,:));
w0 = w0 / norm(w0);

% Objective function
logfun = @(w) sum(log(1+exp(X*w))) + lambda * w'*w;


cf.b = 0;