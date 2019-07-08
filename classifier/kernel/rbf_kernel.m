function out = rbf_kernel(param, x, y)
% Radial basis function (RBF) kernel. 
% 
% Usage:
% out = rbf_kernel(param, X)
% out = rbf_kernel(param, x,y)
% 
%Parameters:
% param          - struct with kernel hyperparameter (gamma)
% X              - [samples x features] data matrix 
%             - OR -
% x,y            - two feature vectors or matrices of feature vectors
%
%Output:
% out            - [samples x samples] kernel matrix or, if x and y are
%                  provided, the kernel evaluated for x and y (ker(x,y))

if nargin == 2
    % compute full kernel matrix
    out = exp(-param.gamma * squareform(pdist(x).^2));
else
    % just evaluate kernel for x and y
    out = exp(-param.gamma * pdist2(x,y).^2);
end