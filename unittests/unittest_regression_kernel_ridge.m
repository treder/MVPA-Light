% Regression unit test
%
% Model: kernel ridge

rng(42)
tol = 10e-10;
mf = mfilename;

%% providing kernel matrix directly VS calculating it from scratch should give same result
% Create spiral data
N = 500;
nrevolutions = 2;       % how often each class spins around the zero point
nclasses = 1;
scale = 0.001;
prop = [];
[X,~] = simulate_spiral_data(N, nrevolutions, nclasses, prop, scale, 0);

% as response variable we choose the Euclidean distance to the origin
y = sqrt(sum(X.^2 ,2 ));

% Get classifier params
param = mv_get_hyperparameter('kernel_ridge');
param.gamma     = 1;

% 1 -provide precomputed kernel matrix
K = rbf_kernel(struct('gamma',1),X);
param.kernel = 'precomputed';
model_kernel = train_kernel_ridge(param, K, y);

% 2 - do not provide kernel matrix (it is calculated in train_kernel_fda)
param.kernel = 'rbf';
model_nokernel = train_kernel_ridge(param, X, y);

% Are all returned values between 0 and 1?
print_unittest_result('providing kernel matrix vs calculating it from scratch should be equal', model_nokernel.alpha, model_kernel.alpha, tol);

%% linear kernel should give the same result as ridge regression
param_ridge = mv_get_hyperparameter('ridge');
param = mv_get_hyperparameter('kernel_ridge');
param.kernel = 'linear';

model_ridge = train_ridge(param_ridge, X, y);
model = train_kernel_ridge(param, X, y);

print_unittest_result('linear kernel should return same w as ridge regression', model_ridge.w, model.w, tol);
print_unittest_result('linear kernel should return same b as ridge regression', model_ridge.b, model.b, tol);

%% increasing lambda should 'shrink' the weights
% (in that norm of w vector decreases as lambda increases. Note that
%  individual entries of w can both increase or decrease)
lambdas = 1:2:20;

nor = zeros(numel(lambdas),1);   % norms of w's
for ii=1:numel(lambdas)
    param.lambda = lambdas(ii);
    model = train_kernel_ridge(param, X, y);
    nor(ii) = norm(model.w);
end

d = diff(nor);
print_unittest_result('increasing lambda shrinks w', all(d<0), true, tol);

%% increasing lambda should increase MSE on TRAIN data (training fit gets worse)
lambdas = linspace(0.1, 10, 20);

perf = zeros(numel(lambdas),1);   % norms of w's
for ii=1:numel(lambdas)
    param.lambda = lambdas(ii);
    model = train_kernel_ridge(param, X, y);
    yhat = test_kernel_ridge(model, X);
    perf(ii) = mv_calculate_performance('mse', '', yhat, y);
end

d = diff(perf);
print_unittest_result('increasing lambda increases MSE on TRAIN set', true, all(d>0), tol);

%% tuning lambda (just run it to make sure it doesn't crash)
param.lambda    = [10^-3, 10^-2, 10^-1, 1, 10, 10, 10^3, 10^4, 10^5, 10^6];
param.gamma     = [0.1, 1, 10, 100];
param.kernel    = 'rbf';
train_kernel_ridge(param, X, y);


%% correlation_bound: checking whether target-residual correlation corresponds to correlation bound
param = mv_get_hyperparameter('kernel_ridge');

param.correlation_bound = 0;
model = train_kernel_ridge(param, X, y);
yhat = test_kernel_ridge(model, X);
print_unittest_result('correlation_bound = 0', 0, corr(y, y - yhat), tol);

param.correlation_bound = 0.1;
model = train_kernel_ridge(param, X, y);
yhat = test_kernel_ridge(model, X);
print_unittest_result('correlation_bound = 0.1', 0.1, corr(y, y - yhat), tol);

param.correlation_bound = 0.2;
model = train_kernel_ridge(param, X, y);
yhat = test_kernel_ridge(model, X);
print_unittest_result('correlation_bound = 0.2', 0.2, corr(y, y - yhat), tol);

param.correlation_bound = 0.9;
model = train_kernel_ridge(param, X, y);
yhat = test_kernel_ridge(model, X);
print_unittest_result('correlation_bound = 0.9', true, corr(y, y - yhat)<=0.9, tol);

%% multivariate Y: predictions should be equal when columns of Y are equal
Y = X * rand(2, 10) +10.01*randn(N,10);       % multivariate model

param = mv_get_hyperparameter('kernel_ridge');

% repeat the first 
model = train_kernel_ridge(param, X, [Y(:,1) Y]);
y_pred = test_kernel_ridge(model, X);

print_unittest_result('[multivariate] same y_pred when columns of Y are the same', y_pred(:,1), y_pred(:,2), tol);
