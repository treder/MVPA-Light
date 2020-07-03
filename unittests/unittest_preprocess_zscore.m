% Preprocessing unit test
%
% zscore

tol = 10e-10;

% Random data
X = randn(6,10,20,30,10) + 1;

% Get default parameters
param = mv_get_preprocess_param('zscore');

%% z-scoring along dimension X should result in all mean=0 and std=1
for dd=1:ndims(X)
    param.dimension = dd;
    [~, Xd] = mv_preprocess_zscore(param, X);
    m = mean(Xd,dd);
    sd = std(Xd,[],dd);
    print_unittest_result(sprintf('[dimension %d] mean should be 0',dd), 0, max(m(:)), tol);
    print_unittest_result(sprintf('[dimension %d] std should be 1',dd), 1, max(sd(:)), tol);
end

%% nested preprocessing: train and test set
tol_train = 10e-8;
tol_test = 1;  % very lenient for test since the scaling is just approximately optimal

X = randn(100, 4)*10 + 100;
Xtrain = X(1:2:end, :);
Xtest = X(2:2:end, :);

pparam = mv_get_preprocess_param('zscore');

% train set
[pparam, Xout_train] = mv_preprocess_zscore(pparam, Xtrain);

% apply on test set
pparam.is_train_set = 0;
[pparam, Xout_test] = mv_preprocess_zscore(pparam, Xtest);

print_unittest_result('[nested preprocessing] mean of X_train is 0', mean(Xout_train,1), zeros(1,size(X,2)), tol_train);
print_unittest_result('[nested preprocessing] mean of X_test is roughly 0', mean(Xout_test,1), zeros(1,size(X,2)), tol_test);
print_unittest_result('[nested preprocessing] std of X_train is 1', std(Xout_train,[],1), ones(1,size(X,2)), tol_train);
print_unittest_result('[nested preprocessing] std of X_test is roughly 1', std(Xout_test,[],1), ones(1,size(X,2)), tol_test);
