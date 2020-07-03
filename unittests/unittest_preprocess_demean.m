% Preprocessing unit test
%
% demean

tol = 10e-10;

% Random data
X = randn(145,40,20,3) - 10;

% Get default parameters
param = mv_get_preprocess_param('demean');

%% demean along dimension X should result in all mean = 0
for dd=1:ndims(X)
    param.dimension = dd;
    [~, Xd] = mv_preprocess_demean(param, X);
    m = mean(Xd,dd);
    print_unittest_result(sprintf('[dimension %d] mean should be 0',dd), 0, max(m(:)), tol);
end

%% nested preprocessing: train and test set
tol_train = 10e-8;
tol_test = 2;  % very lenient for test since the scaling is just approximately optimal

X = randn(1000, 4)*10 + 10;
Xtrain = X(1:2:end, :);
Xtest = X(2:2:end, :);

pparam = mv_get_preprocess_param('demean');

% train set
[pparam, Xout_train] = mv_preprocess_demean(pparam, Xtrain);

% apply on test set
pparam.is_train_set = 0;
[pparam, Xout_test] = mv_preprocess_demean(pparam, Xtest);

print_unittest_result('[nested preprocessing] mean of X_train is 0', mean(Xout_train,1), zeros(1,size(X,2)), tol_train);
print_unittest_result('[nested preprocessing] mean of X_test is roughly 0', mean(Xout_test,1), zeros(1,size(X,2)), tol_test);