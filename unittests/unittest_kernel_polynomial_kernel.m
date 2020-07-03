% kernel unit test
%
% polynomial kernel
tol = 10e-10;

% Random data
N = 10;
F = 5;
X = randn(N,F);
X(1,:) = X(1,:) + X(2,:);

% kernel parameters
param = [];
param.gamma = 1/F;
param.coef0 = 1;
param.degree = 1;

%% polynomial_kernel(X) and polynomial_kernel(X,X) should yield the same result
K1 = polynomial_kernel(param, X);
K2 = polynomial_kernel(param, X, X);

print_unittest_result('polynomial_kernel of (X) and (X,X) should be the same ', 0, norm(K1-K2), tol);

%% quadratic kernel should give all positive entries
param.degree = 2;
K = polynomial_kernel(param, X);

print_unittest_result('quadratic kernel should give all positive entries', 1, all(all(K>0)), tol);

%% gamma=1, coef0=0, degree=1 should yield the linear kernel
param.gamma = 1;
param.coef0 = 0;
param.degree = 1;
K = polynomial_kernel(param, X);
L = linear_kernel([], X);

print_unittest_result('gamma=1, coef0=0, degree=1 should yield the linear kernel', 0, norm(K-L), tol);
