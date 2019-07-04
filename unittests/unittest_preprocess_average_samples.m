% Preprocessing unit test
%
% average_samples

tol = 10e-10;

N = 105;
X = randn(N,1);

% Get default parameters
param = mv_get_preprocess_param('average_samples');

%% if a class has less than group_size samples, it should disappear from the averaged sample
param.group_size = 5;

clabel = ones(N,1);
clabel(N-30:end) = 2;
clabel(N-3:end) = 3;    % less than 5 samples

[~, ~, clabel2] = mv_preprocess_average_samples(param, X, clabel);

print_unittest_result('class with less than group_size samples should have no averages', 0, sum(clabel2==3), tol);

%% number of averages should be equal to N/group_size
% (assumed that group_size is a divisor of the class numbers)
clabel = ones(N,1);
clabel(1:20) = 2;

[~, ~, clabel2] = mv_preprocess_average_samples(param, X, clabel);

print_unittest_result('N/group_size equals number of averages?', N/param.group_size, numel(clabel2), tol);

%% if group_size = 1 data size should be unchanged 
% (though note that the order of the samples changes)
clabel = ones(N,1);
clabel(end-20:end) = 2;
X = randn(numel(clabel),3);

param.group_size = 1;
[~, X2, clabel2] = mv_preprocess_average_samples(param, X, clabel);

print_unittest_result('[.group_size=1] clabel should have equal size', numel(clabel), numel(clabel2), tol);
print_unittest_result('[.group_size=1] X should have equal number of elements', numel(X), numel(X2), tol);

%% .sample_dimension should determine which dimension is averaged
clabel = round(rand(N,1) * 2 + 1);
X = randn(numel(clabel), numel(clabel), 1, numel(clabel));

param.group_size       = 3;
param.sample_dimension = [1,2,4];
[~, X, clabel] = mv_preprocess_average_samples(param, X, clabel);

print_unittest_result('[sample_dimension 1] length of X and clabel match', size(X,1), numel(clabel), tol);
print_unittest_result('[sample_dimension 2] length of X and clabel match', size(X,2), numel(clabel), tol);
print_unittest_result('[sample_dimension 4] length of X and clabel match', size(X,4), numel(clabel), tol);
