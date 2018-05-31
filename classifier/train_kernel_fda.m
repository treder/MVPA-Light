function cf = train_kernel_fda(cfg,X,clabel)
% Trains a kernel Fisher Discriminant Analysis (KFDA). Works with an
% arbitrary number of classes. For a linear kernel, LDA (for two classes)
% or multi-class LDA can be used equivalently.
%
% Usage:
% cf = train_kernel_fda(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% cfg          - struct with hyperparameters:
% kernel         - kernel function:
%                  'rbf'        - radial basis function or Gaussian kernel
%                                 ker(x,y) = exp(-gamma * |x-y|^2);
%                  'polynomial' - polynomial kernel
%                                 ker(x,y) = (gamma * x * y' + coef0)^degree
%                  Alternatively, a custom kernel can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
% kernel_regularisation     - regularisation parameter for the kernel matrix. The
%                  kernel matrix Q is replaced by Q + kernel_reg*I where I
%                  is the identity matrix (default 10e-10)
% Q              - optional kernel matrix. If Q is provided, the .kernel 
%                  parameter is ignored. (Default [])
%
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

% TODO: check the rest of the documentation:

% Further parameters (that usually do not need to be changed):

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

% (c) Matthias Treder 2018

[N, nFeat] = size(X);
nclasses = max(clabel);

%% Set kernel hyperparameter defaults
if ischar(cfg.gamma) && strcmp(cfg.gamma,'auto')
    cfg.gamma = 1/ nFeat;
end

%% Use formulation in https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis#Kernel_trick_with_LDA

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