function model = train_kernel_ridge(param, X, Y)
% Trains a kernel ridge regression model.
%
% Usage:
% cf = train_kernel_fda(param, X, Y)
%
%Parameters:
% X              - [samples x features] matrix of training samples (should
%                                 not include intercept term/column of 1's)
%                  -OR-
%                  [samples x samples] kernel matrix
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
% .kernel        - kernel function:
%                  'linear'     - linear kernel ker(x,y) = x' y
%                  'rbf'        - radial basis function or Gaussian kernel
%                                 ker(x,y) = exp(-gamma * |x-y|^2);
%                  'polynomial' - polynomial kernel
%                                 ker(x,y) = (gamma * x * y' + coef0)^degree
%                  Alternatively, a custom kernel can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
%
%                  If a precomputed kernel matrix is provided as X, set
%                  param.kernel = 'precomputed'.
%
% HYPERPARAMETERS for specific kernels:
%
% gamma         - (kernel: rbf, polynomial) controls the 'width' of the
%                  kernel. If set to 'auto', gamma is set to 1/(nr of features)
%                  (default 'auto')
% coef0         - (kernel: polynomial) constant added to the polynomial
%                 term in the polynomial kernel. If 0, the kernel is
%                 homogenous (default 1)
% degree        - (kernel: polynomial) degree of the polynomial term. A too
%                 high degree makes overfitting likely (default 2)
%
% IMPLEMENTATION DETAILS:
% The solution to the kernel ridge regression problem is given by
%
% alpha = (K + lambda I)^-1  y    (dual form)
%
% where lambda is the regularization hyperparameter and K is the [samples x
% samples] kernel matrix. Predictions on new data are then obtained using 
%
% f(x) = alpha' *  k 
%
% where k = (k(x1, x), ... , k(xn, x))' is the vector of kernel evaluations
% between the training data and the test sample x.
%
% REFERENCE:
% https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/10.pdf

% (c) Matthias Treder

[N, P] = size(X);
model = struct();

% indicates whether kernel matrix has been precomputed
is_precomputed = strcmp(param.kernel,'precomputed');

%% Center X
if strcmp(param.kernel, 'linear')
    model.m = mean(X);
    X = X - repmat(model.m, [N 1]);
end

%% Set kernel hyperparameter defaults
if ischar(param.gamma) && strcmp(param.gamma,'auto') && ~is_precomputed
    param.gamma = 1/size(X,2);
end

%% Check if hyperparameters need to be tuned
tune_params = {};
tune_kernel_params = 0;

if numel(param.lambda)>1, tune_params = [tune_params {'lambda'}]; end
if ~is_precomputed
    if numel(param.gamma)>1, tune_params = [tune_params {'gamma'}]; tune_kernel_params = 1; end
    if numel(param.coef0)>1, tune_params = [tune_params {'coef0'}]; tune_kernel_params = 1; end
    if numel(param.degree)>1, tune_params = [tune_params {'degree'}]; tune_kernel_params = 1; end
end

%% Compute kernel
use_K = 0;
if is_precomputed
    K = X;
    use_K = 1;
elseif tune_kernel_params == 0
    kernelfun = eval(['@' param.kernel '_kernel']);     % Kernel function
    K = kernelfun(param, X);                            % Compute kernel matrix
    use_K = 1;
end

%% Hyperparameter tuning
if ~isempty(tune_params)
    MAE = @(y, ypred) -sum(abs(y - ypred));   % - mean absolute error
    if use_K
        param = mv_tune_hyperparameters(param, K, Y, @train_kernel_ridge, @test_kernel_ridge, ...
            MAE, tune_params, param.k, 1);
    else
        param = mv_tune_hyperparameters(param, X, Y, @train_kernel_ridge, @test_kernel_ridge, ...
            MAE, tune_params, param.k, 0);
    end
    
    % compute kernel with the optimal hyperparameters
    kernelfun = eval(['@' param.kernel '_kernel']);     
    K = kernelfun(param, X);                            
end

%% Perform regularization and calculate weights
alpha = (K + param.lambda * eye(N)) \ Y;

%% Set up model struct
model.alpha         = alpha;
model.lambda        = param.lambda;
model.kernel        = param.kernel;

if strcmp(param.kernel, 'linear')
    model.w = X' * alpha;
    model.b = mean(Y) - model.m * model.w;
else
    if ~is_precomputed
        model.kernelfun    = kernelfun;
        model.X_train      = X;
    end
    
    % Hyperparameters
    model.gamma        = param.gamma;
    model.coef0        = param.coef0;
    model.degree       = param.degree;
    
end

%% Correlation constraint
if ~isempty(param.correlation_bound)
    
    % calculate correlation between y and residual
    yc = Y - mean(Y);
    yhat = K * model.alpha;
    yhatc = yhat - mean(yhat);
    
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
    model.alpha = model.alpha * model.theta;
end