function K = compute_kernel_matrix(cfg, X)
% Given a kernel and data, computes the associated kernel matrix.
% 
% Usage:
% K = compute_kernel_matrix(cfg, X)
% 
%Parameters:
% X              - [samples x features x ...] data matrix
%
% cfg            - [struct] with parameters:
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
%                  Additional fields specifying kernel hyperparameters
%                  should be provided (eg cfg.gamma for the rbf kernel)
%
% .regularize_kernel  - if > 0, an identity matrix multiplied by the value
%                       of regularze_kernel is added to the kernel matrix
%
% If ndims(X)>2 a separate kernel matrix is calculated for each element of
% the other dimensions. This can be useful for constructing a separate
% kernel matrix for e.g. every time point in a trial.
%
%Output:
% K            - [samples x samples x ...] kernel matrix

% Size of X excluding sample/feature dimensions
sz = size(X);
N = sz(1);
sz(1:2) = [];

% Get handle for kernel function 
kernel_fun = eval(['@' cfg.kernel '_kernel']);

% Regularization
if cfg.regularize_kernel > 0
    K = cfg.regularize_kernel * eye(N);
else
    K = zeros(N);
end

if isempty(sz)
    % just one kernel matrix needs to be computed
    K = K + kernel_fun(cfg, X);
else
    % loop through other dimensions and calculate kernel matrices
    K = repmat(K, [1,1,sz]);
    for ix=1:prod(sz)
        K(:,:,ix) = K(:,:,ix) + kernel_fun(cfg, squeeze(X(:,:,ix)));
    end
end

