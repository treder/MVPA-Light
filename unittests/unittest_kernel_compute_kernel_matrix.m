% kernel unit test
%
% compute kernel matrix
tol = 10e-10;

% Random data
N = 30;
F = 10;
X = randn(N, F);

cfg = [];
cfg.kernel              = 'rbf';
cfg.gamma               = 1;
cfg.regularize_kernel   = 1;

%% regularize_kernel should just be adding a diagonal
cfg.regularize_kernel   = 0;
K = compute_kernel_matrix(cfg, X);
cfg.regularize_kernel   = 1;
K_reg = compute_kernel_matrix(cfg, X);

% K_reg - K should be equal to the diagonal I
I = eye(N);

print_unittest_result('regularize_kernel should just be adding a diagonal', 0, norm(K_reg - K - I), tol);

%% check regularization for multi-dimensional input
X = randn(N, F, 1, 2);
lambda = 2;
cfg.regularize_kernel   = lambda;
K = compute_kernel_matrix(cfg, X);

print_unittest_result('check regularization for multi-dimensional input', 0, norm(diag(squeeze(K(:,:,1,1)) - (lambda+1)*I)), tol);

%% check output dimensions for multidimensional input
cfg.regularize_kernel   = 0;

A = 4;
B = 3;
C = 2;
X1 = randn(N, F, A);
X2 = randn(N, F, A, B);
X3 = randn(N, F, A, B, C);
K1 = compute_kernel_matrix(cfg, X1);
K2 = compute_kernel_matrix(cfg, X2);
K3 = compute_kernel_matrix(cfg, X3);

print_unittest_result('[check dimensions] [NxFxA] -> [NxNxA]', 1, all(size(K1) == [N,N,A]), tol);
print_unittest_result('[check dimensions] [NxFxAxB] -> [NxNxAxB]', 1, all(size(K2) == [N,N,A,B]), tol);
print_unittest_result('[check dimensions] [NxFxAxBxC] -> [NxNxAxBxC]', 1, all(size(K3) == [N,N,A,B,C]), tol);

