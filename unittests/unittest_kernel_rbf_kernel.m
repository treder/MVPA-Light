% kernel unit test
%
% rbf kernel
tol = 10e-10;

% Random data
N = 50;
F = 10;
X = randn(N,F);
X(1,:) = X(1,:) + X(2,:);

% kernel parameters
param = [];
param.gamma = 1/F;

%% rbf_kernel(X) and rbf_kernel(X,X) should yield the same result
K1 = rbf_kernel(param, X);
K2 = rbf_kernel(param, X, X);

print_unittest_result('rbf_kernel of (X) and (X,X) should be the same ', 0, norm(K1-K2), tol);

%% decreasing gamma increases off-diagonal elements
K1 = rbf_kernel(struct('gamma',1/F), X);
K2 = rbf_kernel(struct('gamma',1/F/2), X);

print_unittest_result('decreasing gamma increases off-diagonal elements', 1, all(all((K1-K2) <= 0)), tol);

%% setting gamma = 0 should give all 1's
K = rbf_kernel(struct('gamma',0), X);

print_unittest_result('setting gamma = 0 should give all 1''s', 1, all(all(K==1)), tol);

%% setting gamma = very large should give identity matrix
K = rbf_kernel(struct('gamma',100000), X);
I = eye(size(K));

print_unittest_result('setting gamma = large should give all 0''s', 1, all(all((K-I)==0)), tol);
