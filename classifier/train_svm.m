function cf = train_svm(cfg,X,clabel)
% Trains a support vector machine (SVM). The avoid overfitting, the
% classifier weights are penalised using L2-regularisation.
%
% Note: It is recommended to demean (X = demean(X,'constant')) or z-score 
% (X = zscore(X)) the data first to speed up the optimisation.
%
% Usage:
% cf = train_svm(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with hyperparameters:
% C            - regularisation hyperparameter controlling the magnitude
%                  of regularisation. If a single value is given, it is
%                  used for regularisation. If a vector of values is given,
%                  5-fold cross-validation is used to test all the values
%                  in the vector and the best one is selected.
%                  If set to 'auto', a default search grid is used to
%                  automatically determine the best lambda (default
%                  'auto').
% kernel         - kernel function:
%                  'linear'     - linear kernel, trains a linear SVM
%                                 ker(x,y) = x' * y
%                  'rbf'        - radial basis function or Gaussian kernel
%                                 ker(x,y) = exp(-gamma * |x-y|^2);
%                  'polynomial' - polynomial kernel
%                                 ker(x,y) = (gamma * x * y' + coef0)^degree
%                  Alternatively, a custom kernel can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
% Q              - optional kernel matrix. If Q is provided, the .kernel 
%                  parameter is ignored. (Default [])
%
% Note: C is implemented analogous to the classical SVM implementations,
% see libsvm and liblinear. It is roughly reciprocally related to the
% lambda parameter used in LDA/logistic regression, ie C = 1/lambda.
%
% Hyperparameters for specific kernels:
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
% Further parameters (that usually do not need to be changed):
% bias          - if >0 augments the data with a bias term equal to the
%                 value of bias:  X <- [X; bias] and augments the weight
%                 vector with the bias variable w <- [w; b].
%                 If 0, the bias is assumed to be 0. If set to 'auto', the
%                 bias is 1 for linear kernel and 0 for non-linear
%                 kernel (default 'auto'). This is because for non-linear
%                 kernels, a bias is usally not needed (Kecman 2001, p.182)
% K             - the number of folds in the K-fold cross-validation for
%                 the lambda search (default 5)
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

% (c) Matthias Treder 2017

[N, nFeat] = size(X);

% Vector of 1's we need for optimisation
ONE = ones(N,1);

% Make sure labels come as column vector
clabel = double(clabel(:));

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

%% Set kernel hyperparameter defaults
if ischar(cfg.gamma) && strcmp(cfg.gamma,'auto')
    cfg.gamma = 1/ nFeat;
end

%% Bias
if ischar(cfg.bias) && strcmp(cfg.bias,'auto')
    if strcmp(cfg.kernel,'linear')
        cfg.bias = 1;
    else
        cfg.bias = 0;
    end
end

% Augment X with bias
if cfg.bias > 0
    X = cat(2,X, ones(N,1) * cfg.bias );
    nFeat = nFeat + 1;
end


%% Precompute and regularise kernel

if isempty(cfg.Q)
    
    % Kernel function
    kernelfun = eval(['@' cfg.kernel '_kernel']);
    
    % Compute kernel matrix
    Q = kernelfun(cfg, X);
    
    % Regularise
    if cfg.regularise_kernel > 0
        Q = Q + cfg.regularise_kernel * eye(size(X,1));
    end
    
else
    Q = cfg.Q;
end

% Create a copy of Q wherein class labels 1 and -1 are absorbed in the
% kernel matrix. This is useful for optimisation.
Q_cl = Q .* (clabel * clabel');

% %% Automatically set the search grid for C
if ischar(cfg.C) && strcmp(cfg.C,'auto')
    cfg.C = logspace(-4,4,10);
end

%% Optimise hyperparameters using nested cross-validation
if numel(cfg.C)>1
    
   tune_hyperparameter_svm
   
else
    % there is just one lambda: no grid search
    best_idx = 1;
end

%% Train classifier on the full training data (using the best lambda)
C = cfg.C(best_idx);

% Solve the dual problem and obtain alpha
alpha = DualCoordinateDescent(Q_cl, C, ONE, cfg.tolerance, cfg.shrinkage_multiplier);

%% Set up classifier struct
cf= [];
cf.kernel = cfg.kernel;
cf.alpha  = alpha;
cf.gamma  = cfg.gamma;
cf.bias   = cfg.bias;

if strcmp(cfg.kernel,'linear')
    % Calculate linear weights w and bias b from alpha
    cf.w = X' * (cf.alpha .* clabel(:));
    
    if cfg.bias > 0
        cf.b = cf.w(end);
        cf.w = cf.w(1:end-1);
        
        % Bias term needs correct scaling 
        cf.b = cf.b * cfg.bias;
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
    
    if cfg.bias > 0
        %%% TODO: this part needs fixing
        cf.b = 0;
        
%         % remove bias part from support vectors
%         cf.support_vectors = cf.support_vectors(:,1:end-1);
    else
        cf.b = 0;
    end
    
    % Save kernel function
    cf.kernelfun = kernelfun;
    
    % Save hyperparameters
    cf.gamma    = cfg.gamma;
    cf.coef0    = cfg.coef0;
    cf.degree   = cfg.degree;
    
end

end