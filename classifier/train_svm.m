<<<<<<< HEAD
function cf = train_svm(param,X,clabel)
% Trains a support vector machine (SVM). 
=======
function cf = train_svm(cfg,X,clabel)
% Trains a support vector machine (SVM). The avoid overfitting, the
% classifier weights are penalised using L2-regularisation.
>>>>>>> preprocess
%
% Note: It is recommended to demean (X = demean(X,'constant')) or z-score 
% (X = zscore(X)) the data first to speed up the optimisation.
%
% Usage:
<<<<<<< HEAD
% cf = train_svm(param,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% param          - struct with hyperparameters:
=======
% cf = train_svm(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples -OR-
%                  [samples x samples] kernel matrix
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with hyperparameters:
>>>>>>> preprocess
% .c            - regularisation hyperparameter controlling the magnitude
%                  of regularisation. If a single value is given, it is
%                  used for regularisation. If a vector of values is given,
%                  cross-validation is used to test all the values
%                  in the vector and the best one is selected.
%                  If set to 'auto', a default search grid is used to
<<<<<<< HEAD
%                  automatically determine the best c (default
%                  'auto').
% .kernel        - kernel function:
=======
%                  automatically determine the best lambda (default
%                  'auto').
% kernel         - kernel function:
>>>>>>> preprocess
%                  'linear'     - linear kernel, trains a linear SVM
%                                 ker(x,y) = x' * y
%                  'rbf'        - radial basis function or Gaussian kernel
%                                 ker(x,y) = exp(-gamma * |x-y|^2);
%                  'polynomial' - polynomial kernel
%                                 ker(x,y) = (gamma * x' * y + coef0)^degree
<<<<<<< HEAD
%                  Alternatively, a custom kernel can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
% .kernel_matrix - optional kernel matrix. If provided, the .kernel 
%                  parameter is ignored. (Default [])
=======
%                  Alternatively, a custom kernel function can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
%
%                  To provide a precomputed kernel, set cfg.kernel = 'precomputed'.
%
>>>>>>> preprocess
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
<<<<<<< HEAD
=======
%                  or 1 if a pre-computed kernel matrix is provided
>>>>>>> preprocess
%                  (default 'auto')
% coef0         - (kernel: polynomial) constant added to the polynomial
%                 term in the polynomial kernel. If 0, the kernel is
%                 homogenous (default 1)
% degree        - (kernel: polynomial) degree of the polynomial term. A too
%                 high degree makes overfitting likely (default 2)
%
% TUNING: Hyperparameters can be tuned by setting a range instead of a
<<<<<<< HEAD
% single value. For instance, if param.gamma = [10e-1, 1, 10e1] a
=======
% single value. For instance, if cfg.gamma = [10e-1, 1, 10e1] a
>>>>>>> preprocess
% cross-validation is performed where each of the parameters is tested and
% the best parameter is chosen. If multiple parameters are set for tuning,
% a multi-dimensional search grid is set up where all combinations of
% parameters are tested.
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
% plot          - if a lambda search is performed, produces diagnostic
%                 plots including the regularisation path and
%                 cross-validated accuracy as a function of lambda (default
%                 0)
%
%Further parameters controlling the Dual Coordinate Descent optimisation:
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

[N, nFeat] = size(X);

% Vector of 1's we need for optimisation
ONE = ones(N,1);

% Make sure labels come as column vector
clabel = double(clabel(:));

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

%% Set kernel hyperparameter defaults
<<<<<<< HEAD
if ischar(param.gamma) && strcmp(param.gamma,'auto')
    param.gamma = 1/nFeat;
end

%% Bias
if ischar(param.bias) && strcmp(param.bias,'auto')
    if strcmp(param.kernel,'linear')
        param.bias = 1;
    else
        param.bias = 0;
=======
if ischar(cfg.gamma) && strcmp(cfg.gamma,'auto')
    cfg.gamma = 1/nFeat;
end

%% Bias
if ischar(cfg.bias) && strcmp(cfg.bias,'auto')
    if strcmp(cfg.kernel,'linear')
        cfg.bias = 1;
    else
        cfg.bias = 0;
>>>>>>> preprocess
    end
end

% Augment X with bias
<<<<<<< HEAD
if param.bias > 0
    X = cat(2,X, ones(N,1) * param.bias );
=======
if cfg.bias > 0
    X = cat(2,X, ones(N,1) * cfg.bias );
>>>>>>> preprocess
    nFeat = nFeat + 1;
end

%% Check if hyperparmeters need to be tuned
<<<<<<< HEAD
has_kernel_matrix = ~isempty(param.kernel_matrix);
=======
has_kernel_matrix = ~isempty(cfg.kernel_matrix);
>>>>>>> preprocess

% need to tune hyperparameters?
tune_hyperparameters = {};
if ~has_kernel_matrix

    tmp = {};
<<<<<<< HEAD
    switch(param.kernel)
        case 'rbf'
            if numel(param.gamma)>1, tmp = {'gamma' param.gamma}; end
        case 'polynomial'
            if numel(param.gamma)>1, tmp = {'gamma' param.gamma}; end
            if numel(param.coef0)>1, tmp = {tmp{:}; 'coef0' param.coef0}; end
            if numel(param.degree)>1, tmp = {tmp{:}; 'degree' param.degree}; end
=======
    switch(cfg.kernel)
        case 'rbf'
            if numel(cfg.gamma)>1, tmp = {'gamma' cfg.gamma}; end
        case 'polynomial'
            if numel(cfg.gamma)>1, tmp = {'gamma' cfg.gamma}; end
            if numel(cfg.coef0)>1, tmp = {tmp{:}; 'coef0' cfg.coef0}; end
            if numel(cfg.degree)>1, tmp = {rmp{:}; 'degree' cfg.degree}; end
>>>>>>> preprocess
    end
    tune_hyperparameters = tmp;
end

<<<<<<< HEAD
%% Precompute and regularise kernel
=======
%% compute and regularise kernel
>>>>>>> preprocess

if ~has_kernel_matrix && isempty(tune_hyperparameters)
    % If we do not need to tune hyperparameters, we can precompute the
    % kernel matrix

    % Kernel function
<<<<<<< HEAD
    kernelfun = eval(['@' param.kernel '_kernel']);
    
    % Compute kernel matrix
    kernel_matrix = kernelfun(param, X);
    
    % Regularise
    if param.regularise_kernel > 0
        kernel_matrix = kernel_matrix + param.regularise_kernel * eye(size(X,1));
=======
    kernelfun = eval(['@' cfg.kernel '_kernel']);
    
    % Compute kernel matrix
    kernel_matrix = kernelfun(cfg, X);
    
    % Regularise
    if cfg.regularise_kernel > 0
        kernel_matrix = kernel_matrix + cfg.regularise_kernel * eye(size(X,1));
>>>>>>> preprocess
    end
    
else
    % kernel matrix has been provided by the user, so we take it from 
<<<<<<< HEAD
    % the param struct
    kernel_matrix = param.kernel_matrix;
=======
    % the cfg struct
    kernel_matrix = cfg.kernel_matrix;
>>>>>>> preprocess
end

% Create a copy of kernel_matrix wherein class labels 1 and -1 are absorbed in the
% kernel matrix. This is useful for optimisation.
Q_cl = kernel_matrix .* (clabel * clabel');

% %% Automatically set the search grid for c
<<<<<<< HEAD
if ischar(param.c) && strcmp(param.c,'auto')
    param.c = logspace(-4,4,10);
end

%% Optimise hyperparameters using nested cross-validation
if numel(param.c)>1 || ~isempty(tune_hyperparameters)
=======
if ischar(cfg.c) && strcmp(cfg.c,'auto')
    cfg.c = logspace(-4,4,10);
end

%% Optimise hyperparameters using nested cross-validation
if numel(cfg.c)>1 || ~isempty(tune_hyperparameters)
>>>>>>> preprocess
    
   tune_hyperparameter_svm
   
else
    % there is just one lambda: no grid search
    best_c_idx = 1;
end

%% Train classifier on the full training data (using the best lambda)
<<<<<<< HEAD
c = param.c(best_c_idx);

% Solve the dual problem and obtain alpha
alpha = DualCoordinateDescent(Q_cl, c, ONE, param.tolerance, param.shrinkage_multiplier);

%% Set up classifier struct
cf= [];
cf.kernel = param.kernel;
cf.alpha  = alpha;
cf.gamma  = param.gamma;
cf.bias   = param.bias;
cf.prob   = param.prob;

if strcmp(param.kernel,'linear')
    % Calculate linear weights w and bias b from alpha
    cf.w = X' * (cf.alpha .* clabel(:));
    
    if param.bias > 0
=======
c = cfg.c(best_c_idx);

% Solve the dual problem and obtain alpha
alpha = DualCoordinateDescent(Q_cl, c, ONE, cfg.tolerance, cfg.shrinkage_multiplier);

%% Set up classifier struct
cf= [];
cf.kernel = cfg.kernel;
cf.alpha  = alpha;
cf.gamma  = cfg.gamma;
cf.bias   = cfg.bias;
cf.prob   = cfg.prob;

if strcmp(cfg.kernel,'linear')
    % Calculate linear weights w and bias b from alpha
    cf.w = X' * (cf.alpha .* clabel(:));
    
    if cfg.bias > 0
>>>>>>> preprocess
        cf.b = cf.w(end);
        cf.w = cf.w(1:end-1);
        
        % Bias term needs correct scaling 
<<<<<<< HEAD
        cf.b = cf.b * param.bias;
=======
        cf.b = cf.b * cfg.bias;
>>>>>>> preprocess
    else
        cf.b = 0;
    end
else
    % Nonlinear kernel: also need to save support vectors
    cf.support_vector_indices = find(cf.alpha>0);
    cf.support_vectors  = X(cf.support_vector_indices,:);
    
    % Class labels for the support vectors
    cf.y                = clabel(cf.support_vector_indices);
    
    % For convenience we save the product alpha * y for the support vectors
    cf.alpha_y = cf.alpha(cf.support_vector_indices) .* cf.y(:);
    
<<<<<<< HEAD
    if param.bias > 0
=======
    if cfg.bias > 0
>>>>>>> preprocess
        %%% TODO: this part might need fixing
        cf.b = 0;
        
%         % remove bias part from support vectors
%         cf.support_vectors = cf.support_vectors(:,1:end-1);
    else
        cf.b = 0;
    end
    
    cf.has_kernel_matrix = has_kernel_matrix;
    if ~has_kernel_matrix
        cf.kernelfun = kernelfun;
    end
    
    % Save hyperparameters
<<<<<<< HEAD
    cf.gamma    = param.gamma;
    cf.coef0    = param.coef0;
    cf.degree   = param.degree;
=======
    cf.gamma    = cfg.gamma;
    cf.coef0    = cfg.coef0;
    cf.degree   = cfg.degree;
>>>>>>> preprocess
    
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
        dval = kernel_matrix(:, cf.support_vector_indices) * cf.alpha_y   + cf.b;
    end
    
    % Invoke platt function to get sigmoid parameters
    [cf.A, cf.B] = platt(dval, clabel, prior0, prior1);
end

end