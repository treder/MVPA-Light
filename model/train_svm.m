function cf = train_svm(param,X,clabel)
% Trains a support vector machine (SVM). 
%
% Note: It is recommended to demean (X = demean(X,'constant')) or z-score 
% (X = zscore(X)) the data first to speed up the optimization.
%
% Usage:
% cf = train_svm(param,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples -OR-
%                  [samples x samples] kernel matrix
% clabel         - [samples x 1] vector of class labels
%
% param          - struct with hyperparameters:
% .c            - regularization hyperparameter controlling the magnitude
%                  of regularization. If a single value is given, it is
%                  used for regularization. If a vector of values is given,
%                  cross-validation is used to test all the values
%                  in the vector and the best one is selected.
%                  If set to 'auto', a default search grid is used to
%                  automatically determine the best c (default
%                  'auto').
% .kernel        - kernel function:
%                  'linear'     - linear kernel, trains a linear SVM
%                                 ker(x,y) = x' * y
%                  'rbf'        - radial basis function or Gaussian kernel
%                                 ker(x,y) = exp(-gamma * |x-y|^2);
%                  'polynomial' - polynomial kernel
%                                 ker(x,y) = (gamma * x' * y + coef0)^degree
%                  Alternatively, a custom kernel can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
%
%                  If a precomputed kernel matrix is provided as X, set
%                  param.kernel = 'precomputed'.
%
% .prob          - if 1, decision values are returned as probabilities. If
%                  0, the decision values are simply the distance to the
%                  hyperplane. Calculating probabilities takes more time
%                  and memory so don't use this unless needed. A Platt
%                  approximation using an external function (obtained from
%                  http://www.work.caltech.edu/~htlin/program/libsvm/) is
%                  used to estimated probabilities.
%
% Note: c is implemented analogous to the classical SVM implementations,
% see libsvm and liblinear. It is roughly reciprocally related to the
% lambda parameter used in LDA/logistic regression, ie c = 1/lambda.
%
% HYPERPARAMETERS for specific kernels:
%
% gamma         - (kernel: rbf, polynomial) controls the 'width' of the
%                  kernel. If set to 'auto', gamma is set to 1/(nr of features)
%                  or 1 if a pre-computed kernel matrix is provided
%                  (default 'auto')
% coef0         - (kernel: polynomial) constant added to the polynomial
%                 term in the polynomial kernel. If 0, the kernel is
%                 homogenous (default 1)
% degree        - (kernel: polynomial) degree of the polynomial term. A too
%                 high degree makes overfitting likely (default 2)
%
% TUNING: Hyperparameters can be tuned by setting a range instead of a
% single value. For instance, if param.gamma = [10e-1, 1, 10e1] a
% cross-validation is performed where each of the parameters is tested and
% the best parameter is chosen. If multiple parameters are set for tuning,
% a multi-dimensional search grid is set up where all combinations of
% parameters are tested. Note: if a precomputed kernel is provided, only c
% can be tuned.
%
% Further parameters (that usually do not need to be changed):
% bias          - if >0 augments the data with a bias term equal to the
%                 value of bias:  X <- [X; bias] and augments the weight
%                 vector with the bias variable w <- [w; b].
%                 If 0, the bias is assumed to be 0. If set to 'auto', the
%                 bias is 1 for linear kernel and 0 for non-linear
%                 kernel (default 'auto'). This is because for non-linear
%                 kernels, a bias is usally not needed (Kecman 2001, p.182)
% k             - the number of folds in the k-fold cross-validation for
%                 the lambda search (default 3)
% plot          - if a grid search for c is performed, produces diagnostic
%                 plots including the regularization path and
%                 cross-validated accuracy as a function of c (default
%                 0)
%
%Further parameters controlling the Dual Coordinate Descent optimization:
% shrinkage_multiplier  - multiplier that controls the Dual Coordinate
%                 Descent algorithm with active set strategy. If the
%                 multiplier is < 1, the active set is shrunk more
%                 aggressively, potentially leading to speed ups (default
%                 1)
% tolerance     - controls the stopping criterion. If the change in
%                 projected gradient is below tolerance, the algorithm
%                 terminates
%
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - normal to the hyperplane (for linear SVM)
% b            - bias term, setting the threshold  (for linear SVM with bias)
% alpha        - dual vector specifying the weighting of training samples
%                for evaluating the classifier. For alpha > 0 a particular
%                sample is a support vector (for kernel SVM)
%
% IMPLEMENTATION DETAILS:
% A Dual Coordinate Descent algorithm is used to find the optimal alpha
% (weights on the samples), implemented in DualCoordinateDescent.
%
% REFERENCES:
% V Kecman (2001). Learning and Soft Computing: Support Vector Machines, 
% Neural Networks, and Fuzzy Logic Models. MIT Press
%
% Lin, Lin & Weng (2007). A note on Platt's probabilistic outputs for 
% support vector machines. Machine Learning, 68(3), 267-276

% (c) Matthias Treder

N = size(X,1);

% Vector of 1's we need for optimization
ONE = ones(N,1);

% Make sure labels come as column vector
clabel = double(clabel(:));

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

% indicates whether kernel matrix has been precomputed
is_precomputed = strcmp(param.kernel,'precomputed');

%% Set kernel hyperparameter defaults
if ischar(param.gamma) && strcmp(param.gamma,'auto') && ~is_precomputed
    param.gamma = 1/size(X,2);
end

%% Set search grid for c
if ischar(param.c) && strcmp(param.c,'auto')
    param.c = logspace(-2,2,5);
end

%% Bias
if ischar(param.bias) && strcmp(param.bias,'auto')
    if strcmp(param.kernel,'linear')
        param.bias = 1;
    else
        param.bias = 0;
    end
end

% Augment X with bias
if param.bias > 0
    X = cat(2,X, ones(N,1) * param.bias );
end

%% Check if kernel hyperparameters need to be tuned
tune_kernel_parameters = {};

if ~is_precomputed

    % Kernel function
    kernelfun = eval(['@' param.kernel '_kernel']);

    tmp = {};
    switch(param.kernel)
        case 'rbf'
            if numel(param.gamma)>1, tmp = {'gamma' param.gamma}; end
        case 'polynomial'
            if numel(param.gamma)>1, tmp = {'gamma' param.gamma}; end
            if numel(param.coef0)>1, tmp = [tmp(:)' {'coef0'} {param.coef0}]; end
            if numel(param.degree)>1, tmp = [tmp(:)' {'degree'} {param.degree}]; end
    end
    tune_kernel_parameters = tmp;
end

%% Compute and regularize kernel
if isempty(tune_kernel_parameters)
    if is_precomputed
        K = X;
    else
        % Compute kernel matrix
        K = kernelfun(param, X);
    end

    % Regularize
    if param.regularize_kernel > 0
        K = K + param.regularize_kernel * eye(size(X,1));
    end
    
    % Create a copy of K wherein class labels 1 and -1 are absorbed in the
    % kernel matrix. This is useful for optimization.
    Q = K .* (clabel * clabel');
end

%% Optimize hyperparameters using nested cross-validation
if numel(param.c)>1 || ~isempty(tune_kernel_parameters)
   tune_hyperparameter_svm
end

%% Train classifier on the full training data

% Solve the dual problem and obtain alpha
alpha = DualCoordinateDescent(Q, param.c, ONE, param.tolerance, param.shrinkage_multiplier);

%% Set up classifier struct
cf= [];
cf.kernel = param.kernel;
cf.c      = param.c;
cf.alpha  = alpha;
cf.gamma  = param.gamma;
cf.bias   = param.bias;
cf.prob   = param.prob;

if strcmp(param.kernel,'linear')
    % Calculate linear weights w and bias b from alpha
    cf.w = X' * (cf.alpha .* clabel(:));
    
    if param.bias > 0
        cf.b = cf.w(end);
        cf.w = cf.w(1:end-1);
        
        % Bias term needs correct scaling 
        cf.b = cf.b * param.bias;
    else
        cf.b = 0;
    end
else
    % Nonlinear kernel: also need to save support vectors
    cf.support_vector_indices = find(cf.alpha>0);
    
    % Class labels for the support vectors
    cf.y                = clabel(cf.support_vector_indices);
    
    % For convenience we save the product alpha * y for the support vectors
    cf.alpha_y = cf.alpha(cf.support_vector_indices) .* cf.y(:);
    
    if param.bias > 0
        %%% TODO: this part might need checking
        cf.b = 0;
        
%         % remove bias part from support vectors
%         cf.support_vectors = cf.support_vectors(:,1:end-1);
    else
        cf.b = 0;
    end
    
    if ~is_precomputed
        cf.support_vectors = X(cf.support_vector_indices,:); 
        cf.kernelfun = kernelfun;
    end
    
    % Save hyperparameters
    cf.gamma    = param.gamma;
    cf.coef0    = param.coef0;
    cf.degree   = param.degree;
    
end

if cf.prob == 1
    % Invoke external function platt.m. It calculates parameters A and B of
    % the sigmoid that models the probabilities. The method is known as 
    % Platt's approximation
    prior0 = sum(clabel == -1); % prior0: number of negative points
    prior1 = sum(clabel == 1);  % prior1: number of positive points
    
    % Calculate decision values for training data
    if strcmp(cf.kernel,'linear')
        dval = X*cf.w + cf.b;
    else
        dval = K(:, cf.support_vector_indices) * cf.alpha_y   + cf.b;
    end
    
    % Invoke platt function to get sigmoid parameters
    [cf.A, cf.B] = platt(dval, clabel, prior0, prior1);
end

end