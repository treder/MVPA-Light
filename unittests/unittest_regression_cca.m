% Regression unit test
%
% Model: cca

rng(42)
tol = 10e-10;
mf = mfilename;

%% lambda = 0 results should be same as matlab's canoncorr
N = 1000;
n_x = 15;
n_y = 6;

cfg = [];
cfg.n_sample = 1;
cfg.n_channel = n_x;
cfg.n_time_point = N;
cfg.n_narrow   = 5;

X = squeeze(simulate_oscillatory_data(cfg))';
% X = randn(N, n_x);
Y = randn(N, n_y);
Y = Y + 0.05 * X * randn(n_x, n_y);

% matlab version
[Xw,Yw,R,Xv,Yv,STATS] = canoncorr(X,Y);

% cca params
param = mv_get_hyperparameter('cca');
param.lambda_x = 0;
param.lambda_y = 0;
model = train_cca(param, X, Y);

% Are all returned values between 0 and 1?
print_unittest_result('[matlab vs train_cca] same correlation values', R(:), model.r(:), tol);
print_unittest_result('[matlab vs train_cca] same correlation values', R(:), model.r(:), tol);

%% --- rest taken from kernel_regression --

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
