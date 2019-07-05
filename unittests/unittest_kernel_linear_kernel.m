% kernel unit test
%
% linear kernel
tol = 10e-10;

% Random data
N = 20;
F = 15;
X = randn(N,F);
X(1,:) = X(1,:) + X(2,:);

%% linear_kernel(X) and linear_kernel(X,X) should yield the same result
K1 = linear_kernel([], X);
K2 = linear_kernel([], X, X);

print_unittest_result('linear_kernel of (X) and (X,X) should be the same ', 0, norm(K1-K2), tol);
