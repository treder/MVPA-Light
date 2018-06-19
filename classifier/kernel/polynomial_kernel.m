function out = polynomial_kernel(param, x, y)
% Polynomial kernel given by ker(x,y) = (x' * y + c)^d, where x and y are 
% the input vectors and c and d are hyperparameters.
% 
% Usage:
% out = polynomial_kernel(param, X)
% out = polynomial_kernel(param, x,y)
% 
%Parameters:
% param          - struct with kernel hyperparameters (gamma, coef0,
%                  degree)
% X              - [samples x features] data matrix 
%             - OR -
% x,y            - two feature vectors or matrices of feature vectors
%
%Output:
% out            - [samples x samples] kernel matrix or, if x and y are
%                  provided, the kernel evaluated for x and y (ker(x,y))

if nargin == 2
    % compute full kernel matrix
    out = (param.gamma * (x * x') + param.coef0).^param.degree;
else
    % evaluate kernel for x and y
    out = (param.gamma * (x * y') + param.coef0).^param.degree;
end