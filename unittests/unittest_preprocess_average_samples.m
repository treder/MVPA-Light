% Preprocessing unit test
%
% average_samples

tol = 10e-10;

N = 256;
X = randn(N,1);

% Get default parameters
pparam = mv_get_preprocess_param('average_samples');

%% if a class has less than group_size samples, it should disappear from the averaged sample
pparam.group_size = 5;

clabel = ones(N,1);
clabel(N-30:end) = 2;
clabel(N-3:end) = 3;    % less than 5 samples

[~, ~, clabel2] = mv_preprocess_average_samples(pparam, X, clabel);

print_unittest_result('class with less than group_size samples should have no averages', 0, sum(clabel2==3), tol);

%% number of averages should be equal to N/group_size
% (assumed that group_size is a divisor of the class numbers)
clabel = ones(N,1);
clabel(1:2:end) = 2;

pparam.group_size = 4;
[~, ~, clabel2] = mv_preprocess_average_samples(pparam, X, clabel);

print_unittest_result('N/group_size equals number of averages?', N/pparam.group_size, numel(clabel2), tol);

%% if group_size = 1 data size should be unchanged 
% (though note that the order of the samples changes)
clabel = ones(N,1);
clabel(end-20:end) = 2;
X = randn(numel(clabel),3);

pparam.group_size = 1;
[~, X2, clabel2] = mv_preprocess_average_samples(pparam, X, clabel);

print_unittest_result('[.group_size=1] clabel should have equal size', numel(clabel), numel(clabel2), tol);
print_unittest_result('[.group_size=1] X should have equal number of elements', numel(X), numel(X2), tol);

%% .sample_dimension should determine which dimension is averaged >> sample_dimension has been removed
% clabel = round(rand(N,1) * 2 + 1);
% X = randn(numel(clabel), numel(clabel), 1, numel(clabel));
% 
% param.group_size       = 3;
% param.sample_dimension = [1,2,4];
% [~, X, clabel] = mv_preprocess_average_samples(param, X, clabel);
% 
% print_unittest_result('[sample_dimension 1] length of X and clabel match', size(X,1), numel(clabel), tol);
% print_unittest_result('[sample_dimension 2] length of X and clabel match', size(X,2), numel(clabel), tol);
% print_unittest_result('[sample_dimension 4] length of X and clabel match', size(X,4), numel(clabel), tol);


%% nested preprocessing: train and test set [just run to see if there's errors]
clabel = ones(N,1);
clabel(2:2:end) = 2;

Xtrain = X(1:end/2, :);
Xtest = X(end/2+1:end, :);
clabel_train = clabel(1:end/2);
clabel_test = clabel(end/2+1:end);

pparam = mv_get_preprocess_param('average_samples');
pparam.group_size = 2;

% train set
[pparam, ~, clabel_train2] = mv_preprocess_average_samples(pparam, Xtrain, clabel_train);

% apply on test set
pparam.is_train_set = 0;
[pparam, Xout, clabel_test2] = mv_preprocess_average_samples(pparam, Xtest, clabel_test);

print_unittest_result('[nested preprocessing] train set size of clabel', numel(clabel_train)/pparam.group_size, numel(clabel_train2), tol);
print_unittest_result('[nested preprocessing] test set size of clabel', numel(clabel_test)/pparam.group_size, numel(clabel_test2), tol);

