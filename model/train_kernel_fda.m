function cf = train_kernel_fda(param,X,clabel)
% Trains a kernel Fisher Discriminant Analysis (KFDA). Works with an
% arbitrary number of classes. For a linear kernel, it is equivalent to
% LDA.
%
% Usage:
% cf = train_kernel_fda(param,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples -OR-
%                  [samples x samples] kernel matrix
% clabel         - [samples x 1] vector of class labels
%
% param          - struct with hyperparameters:
% .reg          - type of regularization
%                 'shrink': shrinkage regularization using (1-lambda)*N +
%                          lambda*nu*I, where nu = trace(N)/P and P =
%                          number of samples. nu assures that the trace of
%                          N is equal to the trace of the regularization
%                          term. 
%                 'ridge': ridge-type regularization of N + lambda*I,
%                          where N is the dual within-class scatter matrix 
%                          and I is the identity matrix
%                  (default 'shrink')
% .lambda        - if reg='shrink', the regularization parameter ranges 
%                  from 0 to 1 (where 0=no regularization and 1=maximum
%                  regularization). (default 10^-5)
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
% The notation in Mika et al is used below, see also wikipedia page:
% https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis#Kernel_trick_with_LDA
%
% REFERENCE:
% Mika S, Raetsch G, Weston J, Schoelkopf B, Mueller KR (1999). 
% Fisher discriminant analysis with kernels. Neural Networks for Signal
% Processing. IX: 41–48.

% (c) Matthias Treder

% not currently used (since we regularize N):
% kernel_regularization     - regularization parameter for the kernel matrix. The
%                  kernel matrix K is replaced by K + kernel_regularization*I where I
%                  is the identity matrix (default 10e-10)

nclasses = max(clabel);
nsamples = size(X,1);

% Number of samples per class
l = arrayfun(@(c) sum(clabel == c), 1:nclasses);

% indicates whether kernel matrix has been precomputed
is_precomputed = strcmp(param.kernel,'precomputed');

%% Set kernel hyperparameter defaults
if ischar(param.gamma) && strcmp(param.gamma,'auto') && ~is_precomputed
    param.gamma = 1/size(X,2);
end

%% Compute kernel
if is_precomputed
    K = X;
else
    kernelfun = eval(['@' param.kernel '_kernel']);     % Kernel function
    K = kernelfun(param, X);                            % Compute kernel matrix

%     % Regularize
%     if param.regularize_kernel > 0
%         K = K + param.regularize_kernel * eye(size(X,1));
%     end
end

%% N: "Dual" of within-class scatter matrix
N = zeros(nsamples);

% Get indices of samples for each class
ix = arrayfun( @(c) clabel==c, 1:nclasses,'Un',0);

for c=1:nclasses
    N = N + K(:,ix{c}) * (eye(l(c)) - 1/l(c)) * K(ix{c},:);
end

%% Regularization of N
lambda = param.lambda;

if strcmp(param.reg,'shrink')
    % SHRINKAGE REGULARIZATION
    % We write the regularized scatter matrix as a convex combination of
    % the N and an identity matrix scaled to have the same trace as N
    N = (1-lambda)* N + lambda * eye(nsamples) * trace(N)/nsamples;

else
    % RIDGE REGULARIZATION
    % The ridge lambda must be provided directly as a positive number
    N = N + lambda * eye(nsamples);
end

%% M: "Dual" of between-classes scatter matrix

% Get class-wise means
Mj = zeros(nsamples,nclasses);
for c=1:nclasses
    Mj(:,c) = mean( K(:, ix{c}), 2);
end

% Sample mean
Ms = mean(K,2);

% Calculate M
M = zeros(nsamples);
for c=1:nclasses
    M = M + l(c) * (Mj(:,c)-Ms) * (Mj(:,c)-Ms)';
end

%% Calculate A (matrix of alpha's)
[A,~] = eigs( N\M, nclasses-1);

%% Set up classifier struct
cf              = [];
cf.kernel       = param.kernel;
cf.A            = A;
cf.nclasses     = nclasses;

if ~is_precomputed
    cf.kernelfun    = kernelfun;
    cf.Xtrain       = X;
end

% Save projected class centroids
cf.class_means  = Mj'*A;

% Hyperparameters
cf.gamma        = param.gamma;
cf.coef0        = param.coef0;
cf.degree       = param.degree;
    
end
