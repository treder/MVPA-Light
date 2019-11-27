function model = train_ridge(param, X, Y)
% Trains a ridge regression or a linear regression model.
%
% Usage:
% model = train_ridge(param, X, Y)
%
%Parameters:
% X              - [samples x features] matrix of training samples (should
%                                 not include intercept term/column of 1's)
% Y              - [samples x 1] vector of responses (for univariate
%                                regression) -or- 
%                  [samples x m] matrix of responses (for multivariate 
%                                regression with m response variables)
%
% param          - struct with hyperparameters:
% .lambda        - regularization parameter for ridge regression, ranges
%                  from 0 (no regularization) to infinity. For lambda=0,
%                  the model yields standard linear (OLS) regression, for 
%                  lambda > 0 it yields ridge regression (default 1).
% .form          - uses the 'primal' or 'dual' form of the solution to
%                  determine w. If 'auto', auomatically chooses the most 
%                  efficient form depending on whether #samples > #features
%                  (default 'auto').
% .k             - number of cross-validation folds for tuning (default 5)
%
% IMPLEMENTATION DETAILS:
% The classical solution to the ridge regression problem in both the primal
% and the dual form is given by
%
% w = (X' X +  lambda I)^-1 X' y    (primal form)
% w = X' (X X' + lambda I)^-1  y    (dual form)
%
% where lambda is the regularization hyperparameter. For lambda=0 the model
% is equivalent to ordinary linear regression. lambda ranges from 0 to
% infinity.
%
% TUNING:
% The regularization hyperparameter lambda can be tuned using
% nested cross-validation. Tuning is activated when lambda is a vector e.g.
% lambda = [10^-1, 1, 10, 100, 1000]. In this case the param.k controls the
% number of cross-validation folds. The tuning is performed in the script
% tune_hyperparameter_ridge_regression.
%
% REFERENCE: 
% Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani.
% An Introduction to Statistical Learning
%
%Output:
% model - struct specifying the regression model with the following fields:
% w            - weights vector
% b            - intercept

% (c) Matthias Treder

[N, P] = size(X);
model = struct();

%% Center X
model.m = mean(X);
X = X - repmat(model.m, [N 1]);

%% Choose between primal and dual form
if strcmp(param.form, 'auto')
    if N >= P
        form = 'primal';
    else
        form = 'dual';
    end
else
    form = param.form;
end

%% Hyperparameter tuning if necessary
if numel(param.lambda) > 1
    
    % tune hyperparameters using MAE as evaluation function 
    param = mv_tune_hyperparameters(param, X, Y, @train_ridge, @test_ridge, ...
        @(y, ypred) -sum(abs(y - ypred)), {'lambda'}, param.k);
end

model.lambda = param.lambda;

%% Perform regularization and calculate weights
if strcmp(form, 'primal')
    model.w = (X'*X + param.lambda * eye(P)) \ (X' * Y);   % primal
else
    model.w = X' * ((X*X' + param.lambda * eye(N)) \ Y);   % dual
end

%% Estimate intercept
model.b = mean(Y) - model.m * model.w; % m*w makes sure that we do not need to center the test data

