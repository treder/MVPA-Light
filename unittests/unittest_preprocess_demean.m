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
