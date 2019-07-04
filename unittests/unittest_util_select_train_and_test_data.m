% Preprocessing unit test
%

rng(42)
tol = 10e-10;

% Random data
N = 100;
X = randn(N,40,30);
clabel = randi(3, N, 1);

train_indices = [1:N/4, 80:N];
test_indices = setdiff(1:N, train_indices);

cfg = [];
cfg.sample_dimension = 1;
[Xtrain, trainlabel, Xtest, testlabel] = ...
    mv_select_train_and_test_data(cfg, X, clabel, train_indices, test_indices);

%% size of train set should be numel(train_indices)
print_unittest_result('[size train set X] equal to numel(train_indices)', numel(train_indices), size(Xtrain,1), tol);
print_unittest_result('[size train set clabel] equal to numel(train_indices)', numel(train_indices), numel(trainlabel), tol);

%% size of test set should be numel(test_indices)
print_unittest_result('[size test set X] equal to numel(test_indices)', numel(test_indices), size(Xtest,1), tol);
print_unittest_result('[size test set clabel] equal to numel(test_indices)', numel(test_indices), numel(testlabel), tol);

%% cfg.sample_dimension should control the dimension that is subselected
d = 2;
cfg.sample_dimension = d;

train_indices = [1:10, size(X,d)-10:size(X,d)];
test_indices = setdiff(1:size(X,d), train_indices);
[Xtrain, ~, Xtest, ~] = ...
    mv_select_train_and_test_data(cfg, X, clabel, train_indices, test_indices);

print_unittest_result('[size train set X for dimension=2] equal to numel(train_indices)', numel(train_indices), size(Xtrain,d), tol);
print_unittest_result('[size test set X for dimension=2] equal to numel(test_indices)', numel(test_indices), size(Xtest,d), tol);

%% cfg.sample_dimension should control the dimension that is subselected
d = 3;
cfg.sample_dimension = d;

train_indices = [1:10, size(X,d)-10:size(X,d)];
test_indices = setdiff(1:size(X,d), train_indices);
[Xtrain, trainlabel, Xtest, testlabel] = ...
    mv_select_train_and_test_data(cfg, X, clabel, train_indices, test_indices);

print_unittest_result('[size train set X for dimension=3] equal to numel(train_indices)', numel(train_indices), size(Xtrain,d), tol);
print_unittest_result('[size test set X for dimension=3] equal to numel(test_indices)', numel(test_indices), size(Xtest,d), tol);

