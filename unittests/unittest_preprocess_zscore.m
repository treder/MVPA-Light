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
