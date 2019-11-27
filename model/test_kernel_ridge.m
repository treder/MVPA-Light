function y_pred = test_kernel_ridge(model, X)
% Applies an kernel ridge regression model to test data and outputs 
% predicted responses.
% 
% Usage:
% y_hat = test_kernel_ridge(model, X)
% 
%Parameters:
% model          - struct describing the regression model obtained from training 
%                  data.
% X              - [samples x features] matrix of test samples
%
%Output:
% y_pred         - predicted responses

if strcmp(model.kernel,'precomputed')
    y_pred = X * model.alpha_y;
elseif strcmp(model.kernel, 'linear')
    y_pred = X * model.w + model.b;
else
    y_pred = model.kernelfun(model, X, model.X_train) * model.alpha;
end

