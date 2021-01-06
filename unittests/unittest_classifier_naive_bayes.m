% Classifier unit test
%
% Classifier: naive_bayes

tol = 10e-10;
mf = mfilename;

%% test whether probabilities within [0, 1]

%%% Create Gaussian data
nsamples = 60;
nfeatures = 10;
nclasses = 3;
prop = [];
scale = 0.01;
do_plot = 0;

[X_gauss, clabel_gauss] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

% get classifier hyperparameter
param = mv_get_hyperparameter('naive_bayes');

% train and test Naive Bayes classifier
cf = train_naive_bayes(param, X_gauss, clabel_gauss);

[pred, dval, prob] = test_naive_bayes(cf, X_gauss);

% Are all returned values between 0 and 1?
print_unittest_result('[prob] all between 0 and 1',1, all(all((prob >= 0) & (prob <= 1))), tol);

%% check probabilities again [using a different prior]
param.prior = [0.1, 0.9, 0.1];

cf = train_naive_bayes(param, X_gauss, clabel_gauss);
[pred, dval, prob] = test_naive_bayes(cf, X_gauss);

% Are all returned values between 0 and 1?
print_unittest_result('[prob with unequal prior] all between 0 and 1',1, all(all((prob >= 0) & (prob <= 1))), tol);

%% 2D array where last dimension serves as features
param.is_multivariate = 1;
cf = train_naive_bayes(param, X_gauss, clabel_gauss);
[pred, dval, prob] = test_naive_bayes(cf, X_gauss);
print_unittest_result('[is_multivariate=1] pred: size for 2D array, without neighbours', [numel(clabel_gauss) 1], size(pred), tol);
print_unittest_result('[is_multivariate=1] dval: size for 2D array, without neighbours', [numel(clabel_gauss) nclasses], size(dval), tol);
print_unittest_result('[is_multivariate=1] prob: size for 2D array, without neighbours', [numel(clabel_gauss) nclasses], size(prob), tol);

%% 3D array where last dimension serves as features
X = repmat(X_gauss, [1 1 17]);

param.is_multivariate = 1;
param.neighbours = [];
cf = train_naive_bayes(param, X, clabel_gauss);
[pred1, dval1, prob1] = test_naive_bayes(cf, X);
print_unittest_result('[is_multivariate=1] pred: size for 3D array, without neighbours', [numel(clabel_gauss) size(X,2)], size(pred1), tol);
print_unittest_result('[is_multivariate=1] dval: size for 3D array, without neighbours', [numel(clabel_gauss) nclasses size(X,2)], size(dval1), tol);
print_unittest_result('[is_multivariate=1] prob: size for 3D array, without neighbours', [numel(clabel_gauss) nclasses size(X,2)], size(prob1), tol);

param.neighbours = {eye(size(X,2))};
cf = train_naive_bayes(param, X, clabel_gauss);
[pred2, dval2, prob2] = test_naive_bayes(cf, X);
print_unittest_result('[is_multivariate=1] pred: compare with and without neighbours', pred1, pred2, tol);
print_unittest_result('[is_multivariate=1] dval: compare with and without neighbours', dval1, dval2, tol);
print_unittest_result('[is_multivariate=1] prob: compare with and without neighbours', prob1, prob2, tol);

%% 2D array where last dimension does not serve as features, without neighbours
param.is_multivariate = 0;
param.neighbours = [];
cf = train_naive_bayes(param, X_gauss, clabel_gauss);
[pred1, dval1, prob1] = test_naive_bayes(cf, X_gauss);
print_unittest_result('[is_multivariate=0] pred: size for 2D array, without neighbours', [numel(clabel_gauss) size(X,2)], size(pred1), tol);
print_unittest_result('[is_multivariate=0] dval: size for 2D array, without neighbours', [numel(clabel_gauss) nclasses size(X,2)], size(dval1), tol);
print_unittest_result('[is_multivariate=0] prob: size for 2D array, without neighbours', [numel(clabel_gauss) nclasses size(X,2)], size(prob1), tol);

param.neighbours = {eye(size(X,2))};
cf = train_naive_bayes(param, X_gauss, clabel_gauss);
[pred2, dval2, prob2] = test_naive_bayes(cf, X_gauss);
print_unittest_result('[is_multivariate=1] pred: compare with and without neighbours', pred1, pred2, tol);
print_unittest_result('[is_multivariate=1] dval: compare with and without neighbours', dval1, dval2, tol);
print_unittest_result('[is_multivariate=1] prob: compare with and without neighbours', prob1, prob2, tol);

%% 3D array where last dimension does not serve as features, without neighbours
X = repmat(X_gauss, [1 1 18]);

param.is_multivariate = 0;
param.neighbours = [];
cf = train_naive_bayes(param, X, clabel_gauss);
[pred1, dval1, prob1] = test_naive_bayes(cf, X);
print_unittest_result('[is_multivariate=0] pred: size for 3D array, without neighbours', [numel(clabel_gauss) size(X,2) size(X,3)], size(pred1), tol);
print_unittest_result('[is_multivariate=0] dval: size for 3D array, without neighbours', [numel(clabel_gauss) nclasses size(X,2) size(X,3)], size(dval1), tol);
print_unittest_result('[is_multivariate=0] prob: size for 3D array, without neighbours', [numel(clabel_gauss) nclasses size(X,2) size(X,3)], size(prob1), tol);

param.neighbours = {eye(size(X,2)) eye(size(X,3))};
cf = train_naive_bayes(param, X, clabel_gauss);
[pred2, dval2, prob2] = test_naive_bayes(cf, X);
print_unittest_result('[is_multivariate=1] pred: compare with and without neighbours', pred1, pred2, tol);
print_unittest_result('[is_multivariate=1] dval: compare with and without neighbours', dval1, dval2, tol);
print_unittest_result('[is_multivariate=1] prob: compare with and without neighbours', prob1, prob2, tol);

%% neighbours with non-square identity matrix 
X = repmat(X_gauss, [1 1 18]);

param.prior = [.25 .25 .5];

param.is_multivariate = 0;
nb1 = eye(size(X,2)); % identity matrix
nb2 = eye(size(X,3));

param.neighbours = {nb1, nb2};
cf = train_naive_bayes(param, X, clabel_gauss);
[pred1, dval1, prob1] = test_naive_bayes(cf, X);

% remove some rows from neighbours matrix - the results should be the same
% as taking the full neighbours matrix and removing some rows from the
% result
param.neighbours = {nb1(5:end, :), nb2(2:end,:)};
cf = train_naive_bayes(param, X, clabel_gauss);
[pred2, dval2, prob2] = test_naive_bayes(cf, X);
print_unittest_result('non-square neighbours: pred', pred1(:,5:end,2:end), pred2, tol);
print_unittest_result('non-square neighbours: dval', dval1(:,:,5:end,2:end), dval2, tol);
print_unittest_result('non-square neighbours: prob', prob1(:,:,5:end,2:end), prob2, tol);

%% neighbours with non-square random matrix 
X = repmat(X_gauss, [1 1 13]);

param.is_multivariate = 0;
nb1 = double(randn(size(X,2))>0); % random neighbour matrix
nb2 = double(randn(size(X,3))>0);

param.neighbours = {nb1, nb2};
cf = train_naive_bayes(param, X, clabel_gauss);
[pred1, dval1, prob1] = test_naive_bayes(cf, X);

% remove some rows from neighbours matrix - the results should be the same
% as taking the full neighbours matrix and removing some rows from the
% result
param.neighbours = {nb1(3:end, :), nb2(4:end-1,:)};
cf = train_naive_bayes(param, X, clabel_gauss);
[pred2, dval2, prob2] = test_naive_bayes(cf, X);
print_unittest_result('non-square neighbours: pred', pred1(:,3:end,4:end-1), pred2, tol);
print_unittest_result('non-square neighbours: dval', dval1(:,:,3:end,4:end-1), dval2, tol);
print_unittest_result('non-square neighbours: prob', prob1(:,:,3:end,4:end-1), prob2, tol);
