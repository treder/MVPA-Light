function out = linear_kernel(~, x,y)
% Linear kernel given by the standard scalar product, ie ker(x,y) = x' * y.  
% 
% Usage:
% ker = linear_kernel(param, X)
% val = linear_kernel(param, x,y)
% 
%Parameters:
% param          - struct with kernel hyperparameter (none for linear
%                   kernel)
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