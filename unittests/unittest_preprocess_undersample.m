% Preprocessing unit test
%
% undersample

tol = 10e-10;

N = 120;
X = randn(N,1);

% Get default parameters
param = mv_get_preprocess_param('undersample');

%% if data is already balanced then undersampling should have no effect
clabel = ones(N,1);
clabel(N/3+1:2*N/3) = 2;
clabel(2*N/3+1:end) = 3;

[~, ~, clabel2] = mv_preprocess_undersample(param, X, clabel);

print_unittest_result('[balanced data] undersampling should not affect data', numel(clabel2), numel(clabel), tol);

%% after undersample: size of each class equal to minority class
clabel = ones(N,1);
clabel(end-30:end) = 2;
clabel(end-10:end) = 3;

n_minority = min(arrayfun(@(c) sum(clabel==c), 1:max(clabel)));
[~, ~, clabel2] = mv_preprocess_undersample(param, X, clabel);
n_per_class = arrayfun(@(c) sum(clabel2==c), 1:max(clabel2));

print_unittest_result('[number of samples] all classes equal size to minority class', 1, all(n_minority - n_per_class)==0 , tol);

%% X and clabel should be undersampled correspondingly
clabel = ones(N,1);
clabel(end-30:end) = 2;
clabel(end-10:end) = 3;
X = randn(numel(clabel), 1);

[~, X, clabel] = mv_preprocess_undersample(param, X, clabel);

print_unittest_result('[dimension 1] length of X and clabel match', size(X,1), numel(clabel), tol);

%% .sample_dimension should determine which dimension is undersampled
clabel = round(rand(N,1) * 2 + 1);
X = randn(numel(clabel), numel(clabel), 1, numel(clabel));

param.sample_dimension = [1,2,4];
[~, X, clabel] = mv_preprocess_undersample(param, X, clabel);

print_unittest_result('[sample_dimension 1] length of X and clabel match', size(X,1), numel(clabel), tol);
print_unittest_result('[sample_dimension 2] length of X and clabel match', size(X,2), numel(clabel), tol);
print_unittest_result('[sample_dimension 4] length of X and clabel match', size(X,4), numel(clabel), tol);

%% no undersampling if is_train_set = 0
param = mv_get_preprocess_param('undersample');
param.is_train_set = 0;

clabel = round(rand(N,1) * 2 + 1);
X = randn(numel(clabel), 2);

[~, ~, clabel2] = mv_preprocess_undersample(param, X, clabel);

print_unittest_result('[is_train_set=0] size of data should be unchanged', numel(clabel2), numel(clabel), tol);

%% if is_train_set = 0 but undersample_test_set = 1 then undersample should take place
param = mv_get_preprocess_param('undersample');
param.is_train_set = 0;
param.undersample_test_set = 1;

clabel = round(rand(N,1) * 2 + 1);
X = randn(numel(clabel), 2);

[~, X2, clabel2] = mv_preprocess_undersample(param, X, clabel);

print_unittest_result('[is_train_set=0 but undersample test_set=1] size of data should change', 1, numel(clabel2) < numel(clabel), tol);

