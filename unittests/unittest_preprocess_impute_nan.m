% Preprocessing unit test
%
% impute_nan


%% TODO

tol = 10e-10;

% Get default parameters
param = mv_get_preprocess_param('impute_nan');

%% simple example for single nan: test whether forward impute works
param.impute_dimension = 1;

X = randn(3,5);
clabel = ones(3,1);
X(1,3) = nan;

[~, X2, ~] = mv_preprocess_undersample(param, X, clabel);

print_unittest_result('[foward for single nan] impute dim=1', X(2,5), X(3,5), tol);

