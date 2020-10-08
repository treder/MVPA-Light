% Regression unit test
%
% Model: ridge regression

rng(42)
tol = 10e-10;
mf = mfilename;

% Generate regression data
nfeatures = 100;
X = randn(1000, nfeatures);
y = X * rand(nfeatures, 1) + 10.01*randn(1000,1);         % univariate model
Y = X * rand(nfeatures, 10) +10.01*randn(1000,10);       % multivariate model

%% for lambda = 0, the w should be equal to regression weights using the regress function
param = mv_get_hyperparameter('ridge');
param.lambda = 0;

model = train_ridge(param, X, y);
B = regress(y, [X, ones(size(X,1), 1)]);

print_unittest_result('[lambda=0] weights w equal to linear regression weights?',model.w, B(1:end-1), tol);
print_unittest_result('[lambda=0] intercept b equal to linear regression b?',model.b, B(end), tol);

%% increasing lambda should 'shrink' the weights
% (in that norm of w vector decreases as lambda increases. Note that
%  individual entries of w can both increase or decrease)
lambdas = 0:2:20;

nor = zeros(numel(lambdas),1);   % norms of w's
for ii=1:numel(lambdas)
    param.lambda = lambdas(ii);
    model = train_ridge(param, X, y);
    nor(ii) = norm(model.w);
end

d = diff(nor);
print_unittest_result('increasing lambda shrinks w', all(d<0), true, tol);

%% increasing lambda should increase MSE on TRAIN data (training fit gets worse)
lambdas = 0:2:20;

perf = zeros(numel(lambdas),1);   % norms of w's
for ii=1:numel(lambdas)
    param.lambda = lambdas(ii);
    model = train_ridge(param, X, y);
    yhat = test_ridge(model, X);
    perf(ii) = mv_calculate_performance('mse', '', yhat, y);
end

d = diff(perf);
print_unittest_result('increasing lambda increases MSE on TRAIN set', true, all(d>0), tol);

%% tuning lambda (just run it to make sure it doesn't crash)
param.lambda = [10^-3, 10^-2, 10^-1, 1, 10, 10, 10^3, 10^4, 10^5, 10^6];
param.plot = 0;
train_ridge(param, X, y);

%% correlation_bound: checking whether target-residual correlation corresponds to correlation bound
param = mv_get_hyperparameter('ridge');
param.lambda = 0.1;

param.correlation_bound = 0;
model = train_ridge(param, X, y);
yhat = test_ridge(model, X);
print_unittest_result('correlation_bound = 0', 0, corr(y, y - yhat), tol);

param.correlation_bound = 0.1;
model = train_ridge(param, X, y);
yhat = test_ridge(model, X);
print_unittest_result('correlation_bound = 0.1', 0.1, corr(y, y - yhat), tol);

param.correlation_bound = 0.2;
model = train_ridge(param, X, y);
yhat = test_ridge(model, X);
print_unittest_result('correlation_bound = 0.2', 0.2, corr(y, y - yhat), tol);

param.correlation_bound = 0.9;
model = train_ridge(param, X, y);
yhat = test_ridge(model, X);
print_unittest_result('correlation_bound = 0.9', true, corr(y, y - yhat)<=0.9, tol);

%% multivariate Y: weights should be equal when columns of Y are equal
param = mv_get_hyperparameter('ridge');
param.lambda = 0.1;

% repeat the first 
model = train_ridge(param, X, [Y(:,1) Y]);

print_unittest_result('[multivariate] same w when columns of Y are the same', model.w(:,1), model.w(:,2), tol);

%% multivariate Y: check size of matrix of predictions
model = train_ridge(param, X, Y);
Y_hat = test_ridge(model, X);

print_unittest_result('[multivariate] size of Y and Y_hat the same on train data', size(Y), size(Y_hat), tol);
