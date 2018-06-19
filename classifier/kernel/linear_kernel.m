function out = linear_kernel(~,x,y)
% Linear kernel given by the standard scalar product, ie ker(x,y) = x' * y.  
% 
% Usage:
% out = linear_kernel(param, X)
% out = linear_kernel(param, x, y)
% 
%Parameters:
% param          - struct with kernel hyperparameter (there's none for
%                  linear kernels but this way the function interface 
%                  is consistent with the other kernel functions)
% X              - [samples x features] data matrix 
%             - OR -
% x,y            - two feature vectors or matrices of feature vectors
%
%Output:
% out            - [samples x samples] kernel matrix or, if x and y are
%                  provided, the kernel evaluated for x and y (ker(x,y))

if nargin == 2
    % compute full kernel matrix
    out = x * x';
else
    % evaluate kernel for x and y
    out = x * y';
end