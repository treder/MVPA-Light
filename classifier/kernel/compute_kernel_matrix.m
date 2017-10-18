function ker = compute_kernel_matrix(cfg, X)
% Given a kernel and data, computes the associated kernel matrix.
% 
% Usage:
% ker = compute_kernel_matrix(cfg, X)
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
% ker            - [samples x samples] kernel matrix

ker = eval([cfg.kernel '_kernel(cfg,X)']);
