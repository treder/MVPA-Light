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
% .correlation_bound - float in [0, 1] specifying the maximum permissible
%                      corelation between y and the residuals (useful in
%                      e.g. brain-age prediction) (default [])
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

%% Correlation constraint
if ~isempty(param.correlation_bound)
    
    % calculate correlation between y and residual
    yc = Y - mean(Y);
    yhatc = X*model.w;
    y_residual_correlation_uncorrected = corr(yc, yc - yhatc);
    
    if y_residual_correlation_uncorrected < param.correlation_bound
        % dont fix it if it aint broken
        model.theta = 1;
    elseif param.correlation_bound == 0
        model.theta = (yc'*yc) / (yc' * yhatc);
    else
        y2 = yc' * yc;
        yhat2 = yhatc' * yhatc;
        yyhat = yc' * yhatc;
        rho2 = param.correlation_bound^2;
        c = yyhat^2 - rho2*y2*yhat2;

        % since we use a square to solve for a we get two solutions
        % one gives us corr(y,e) = rho, the other corr(y,e) = -rho
        model.theta = y2 * yyhat * (1-rho2)/c - y2/c * sqrt( rho2 * (1-rho2) * (y2*yhat2 - yyhat^2));
    end
    
    % scale coefficients
    model.w = model.w * model.theta;
end

%% Estimate intercept
model.b = mean(Y) - model.m * model.w; % m*w makes sure that we do not need to center the test data

