function K = compute_kernel_matrix(cfg, X)
% Given a kernel and data, computes the associated kernel matrix.
% 
% Usage:
% K = compute_kernel_matrix(cfg, X)
% 
%Parameters:
% cfg            - struct that must contain a field .kernel specifying the 
%                  kernel, e.g. 'rbf', 'polynomial'.
%                  There needs to be a function starting with the kernel
%                  name and ending in _kernel (e.g. rbf_kernel). Additional
%                  fields describing hyperparameters (e.g. .gamma for the
%                  RBF kernel) need to be provided.
% X              - [samples x features] data matrix
%
%Output:
% K            - [samples x samples] kernel matrix

% Get handle for kernel function 
kernel_fun = eval(['@' cfg.kernel '_kernel']);

%%% TODO handle multi-dimensional data e.g. [samples x features x time]

% Compute kernel matrix
K = kernel_fun(cfg, X);

% Regularize kernel matrix
if cfg.regularize_kernel > 0
    K = K + cfg.regularize_kernel * eye(size(X,1));
end
