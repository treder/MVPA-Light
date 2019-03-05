function cf = train_kernel_fda(cfg,X,clabel)
% Trains a kernel Fisher Discriminant Analysis (KFDA). Works with an
% arbitrary number of classes. For a linear kernel, it is equivalent to
% LDA (for two classes) or multi-class LDA.
%
% Usage:
% cf = train_kernel_fda(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% cfg          - struct with hyperparameters:
% .reg          - type of regularisation
%                 'shrink': shrinkage regularisation using (1-lambda)*N +
%                          lambda*nu*I, where nu = trace(N)/P and P =
%                          number of samples. nu assures that the trace of
%                          N is equal to the trace of the regularisation
%                          term. 
%                 'ridge': ridge-type regularisation of N + lambda*I,
%                          where N is the dual within-class scatter matrix 
%                          and I is the identity matrix
%                  (default 'shrink')
% .lambda        - if reg='shrink', the regularisation parameter ranges 
%                  from 0 to 1 (where 0=no regularisation and 1=maximum
%                  regularisation). (default 10^-5)
% .kernel        - kernel function:
%                  'linear'     - linear kernel ker(x,y) = x' y
%                  'rbf'        - radial basis function or Gaussian kernel
%                                 ker(x,y) = exp(-gamma * |x-y|^2);
%                  'polynomial' - polynomial kernel
%                                 ker(x,y) = (gamma * x * y' + coef0)^degree
%                  Alternatively, a custom kernel can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
% .kernel_matrix - optional kernel matrix. If provided, the .kernel 
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
% IMPLEMENTATION DETAILS:
% The notation in Mika et al is used below, see also wikipedia page:
% https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis#Kernel_trick_with_LDA
%
% REFERENCE:
% Mika S, Raetsch G, Weston J, Schoelkopf B, Mueller KR (1999). 
% Fisher discriminant analysis with kernels. Neural Networks for Signal
% Processing. IX: 41–48.

% (c) Matthias Treder 2018

% not currently used (since we regularise N):
% kernel_regularisation     - regularisation parameter for the kernel matrix. The
%                  kernel matrix K is replaced by K + kernel_regularisation*I where I
%                  is the identity matrix (default 10e-10)

nclasses = max(clabel);
[nsamples, nfeatures] = size(X);

% Number of samples per class
l = arrayfun(@(c) sum(clabel == c), 1:nclasses);

%% Set kernel hyperparameter defaults
if ischar(cfg.gamma) && strcmp(cfg.gamma,'auto')
    cfg.gamma = 1/ nfeatures;
end

%% Precompute kernel
if isempty(cfg.kernel_matrix)
    
    has_kernel_matrix = 0;
    
    % Kernel function
    kernelfun = eval(['@' cfg.kernel '_kernel']);
    
    % Compute kernel matrix
    kernel_matrix = kernelfun(cfg, X);

else
    has_kernel_matrix = 1;
    kernel_matrix = cfg.kernel_matrix;
end

%% N: "Dual" of within-class scatter matrix
N = zeros(nsamples);
for c=1:nclasses
    N = N + kernel_matrix(:,clabel==c) * (eye(l(c)) - 1/l(c)) * kernel_matrix(clabel==c,:);
end

%% Regularisation of N
lambda = cfg.lambda;

if strcmp(cfg.reg,'shrink')
    % SHRINKAGE REGULARISATION
    % We write the regularised scatter matrix as a convex combination of
    % the N and an identity matrix scaled to have the same trace as N
    N = (1-lambda)* N + lambda * eye(nsamples) * trace(N)/nsamples;

else
    % RIDGE REGULARISATION
    % The ridge lambda must be provided directly as a number
    N = N + lambda * eye(nsamples);
end

%% "Dual" of between-classes scatter matrix

% Get indices of samples for each class
cidx = arrayfun( @(c) clabel==c, 1:nclasses,'Un',0);

% Get class-wise means
Mj = zeros(nsamples,nclasses);
for c=1:nclasses
    Mj(:,c) = mean( kernel_matrix(:, cidx{c}), 2);
end

% Sample mean
Ms = mean(kernel_matrix,2);

% Calculate M
M = zeros(nsamples);
for c=1:nclasses
    M = M + l(c) * (Mj(:,c)-Ms) * (Mj(:,c)-Ms)';
end

%% Calculate A (matrix of alpha's)
[A,~] = eigs( N\M, nclasses-1);

%% Set up classifier struct
cf              = [];
cf.kernel       = cfg.kernel;
cf.A            = A;
cf.nclasses     = nclasses;

cf.has_kernel_matrix = has_kernel_matrix;
if ~has_kernel_matrix
    cf.kernelfun    = kernelfun;
end

% Save training data
cf.Xtrain       = X;

% Save projected class centroids
cf.class_means  = Mj'*A;

% Hyperparameters
cf.gamma        = cfg.gamma;
cf.coef0        = cfg.coef0;
cf.degree       = cfg.degree;
    
end
