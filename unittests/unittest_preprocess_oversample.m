% Preprocessing unit test
%
% oversample

tol = 10e-10;

N = 99;
X = randn(N,1);

% Get default parameters
param = mv_get_preprocess_param('oversample');

%% if data is already balanced then oversampling should have no effect
clabel = ones(N,1);
clabel(N/3+1:2*N/3) = 2;
clabel(2*N/3+1:end) = 3;

[~, ~, clabel2] = mv_preprocess_oversample(param, X, clabel);

print_unittest_result('[balanced data] oversampling should not affect data', numel(clabel2), numel(clabel), tol);

%% after oversample: size of each class equal to majority class
clabel = ones(N,1);
clabel(1:45) = 2;
clabel(end-10:end) = 3;

n_majority = max(arrayfun(@(c) sum(clabel==c), 1:max(clabel)));
[~, ~, clabel2] = mv_preprocess_oversample(param, X, clabel);
n_per_class = arrayfun(@(c) sum(clabel2==c), 1:max(clabel2));

print_unittest_result('[number of samples] all classes equal size to majority class', 1, all(n_majority - n_per_class)==0 , tol);

%% X and clabel should be sampled correspondingly
clabel = ones(N,1);
clabel(1:30) = 2;
clabel(end-10:end) = 3;
X = randn(numel(clabel), 1);

[~, X, clabel] = mv_preprocess_oversample(param, X, clabel);

print_unittest_result('[dimension 1] length of X and clabel match', size(X,1), numel(clabel), tol);

%% .sample_dimension should determine which dimension is sampled
clabel = round(rand(N,1) * 2 + 1);
X = randn(numel(clabel), numel(clabel), 1, numel(clabel));

param.sample_dimension = [1,2,4];
[~, X, clabel] = mv_preprocess_oversample(param, X, clabel);

print_unittest_result('[sample_dimension 1] length of X and clabel match', size(X,1), numel(clabel), tol);
print_unittest_result('[sample_dimension 2] length of X and clabel match', size(X,2), numel(clabel), tol);
print_unittest_result('[sample_dimension 4] length of X and clabel match', size(X,4), numel(clabel), tol);

%% no oversampling if is_train_set = 0
param = mv_get_preprocess_param('oversample');
param.is_train_set = 0;

clabel = round(rand(N,1) * 2 + 1);
X = randn(numel(clabel), 2);

[~, ~, clabel2] = mv_preprocess_oversample(param, X, clabel);

print_unittest_result('[is_train_set=0] size of data should be unchanged', numel(clabel2), numel(clabel), tol);

%% if is_train_set = 0 but oversample_test_set = 1 then undersample should take place
param = mv_get_preprocess_param('oversample');
param.is_train_set = 0;
param.oversample_test_set = 1;

clabel = round(rand(N,1) * 2 + 1);
X = randn(numel(clabel), 2);

[~, X2, clabel2] = mv_preprocess_oversample(param, X, clabel);

print_unittest_result('[is_train_set=0 but oversample test_set=1] size of data should change', 1, numel(clabel2) > numel(clabel), tol);

