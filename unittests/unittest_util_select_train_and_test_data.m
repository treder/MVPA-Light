% util unit test
%
% mv_select_train_and_test_data
tol = 10e-10;

% Random data
N = 100;
ntime = 30;
X = randn(N,40,ntime);
clabel = randi(3, N, 1);

train_indices = [1:N/4, 80:N];
test_indices = setdiff(1:N, train_indices);

n_tr = numel(train_indices);
n_te = numel(test_indices);

is_kernel_matrix = 0;
[Xtrain, trainlabel, Xtest, testlabel] = ...
    mv_select_train_and_test_data(X, clabel, train_indices, test_indices, is_kernel_matrix);

%% size of train set should be numel(train_indices)
print_unittest_result('[size train set X] equal to numel(train_indices)', n_tr, size(Xtrain,1), tol);
print_unittest_result('[size train set clabel] equal to numel(train_indices)', n_tr, numel(trainlabel), tol);

%% size of test set should be numel(test_indices)
print_unittest_result('[size test set X] equal to numel(test_indices)', n_te, size(Xtest,1), tol);
print_unittest_result('[size test set clabel] equal to numel(test_indices)', n_te, numel(testlabel), tol);

%% selecting a kernel both dimensions should be affected
cfg = [];
cfg.kernel = 'rbf';
cfg.gamma  = 1;
cfg.regularize_kernel = 0;

K = compute_kernel_matrix(cfg, X);
is_kernel_matrix = 1;

[Ktrain, trainlabel, Ktest, testlabel] = ...
    mv_select_train_and_test_data(K, clabel, train_indices, test_indices, is_kernel_matrix);

print_unittest_result('[precomputed kernel] train kernel is [train x train samples]', 1, all(size(Ktrain)==[n_tr,n_tr,ntime]), tol);
print_unittest_result('[precomputed kernel] test kernel is [test x train samples]', 1, all(size(Ktest)==[n_te, n_tr,ntime]), tol);


% %% cfg.sample_dimension should control the dimension that is subselected
% d = 2;
% 
% train_indices = [1:10, size(X,d)-10:size(X,d)];
% test_indices = setdiff(1:size(X,d), train_indices);
% [Xtrain, ~, Xtest, ~] = ...
%     mv_select_train_and_test_data(X, clabel, train_indices, test_indices);
% 
% print_unittest_result('[size train set X for dimension=2] equal to numel(train_indices)', numel(train_indices), size(Xtrain,d), tol);
% print_unittest_result('[size test set X for dimension=2] equal to numel(test_indices)', numel(test_indices), size(Xtest,d), tol);
% 
% %% cfg.sample_dimension should control the dimension that is subselected
% d = 3;
% cfg.sample_dimension = d;
% 
% train_indices = [1:10, size(X,d)-10:size(X,d)];
% test_indices = setdiff(1:size(X,d), train_indices);
% [Xtrain, trainlabel, Xtest, testlabel] = ...
%     mv_select_train_and_test_data(X, clabel, train_indices, test_indices);
% 
% print_unittest_result('[size train set X for dimension=3] equal to numel(train_indices)', numel(train_indices), size(Xtrain,d), tol);
% print_unittest_result('[size test set X for dimension=3] equal to numel(test_indices)', numel(test_indices), size(Xtest,d), tol);
% 
