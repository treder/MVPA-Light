% Preprocessing unit test
%
% impute_nan

tol = 10e-10;

% Get default parameters
param = mv_get_preprocess_param('impute_nan');

%% block of Nans at beginning and end of matrix
clabel = ones(5,1);
X = magic(5);
X(1:3,1:2) = nan;
X(end-2:end,end-1:end) = nan;

param.impute_dimension = 1;
param.method = 'forward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward] block of Nans at beginning and end of matrix', true, all(all(isnan(X2(1:3,1:2)))) && all(all(X2(3:5, 4:5) == X(2, 4:5))), tol);

param.method = 'backward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[backward] block of Nans at beginning and end of matrix', true, all(all(isnan(X2(3:5,4:5)))) && all(all(X2(1:3, 1:2) == X(4, 1:2))), tol);

%% simple example for single nan
clabel = ones(4,1);
X = magic(4);
X(2,3) = nan;

param.impute_dimension = 1;
param.method = 'forward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 2D dim=1] filled', X(1,3), X2(2,3), tol);
param.method = 'backward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[backward 2D dim=1] filled', X(3,3), X2(2,3), tol);

param.impute_dimension = 2;
param.method = 'forward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 2D dim=2] filled', X(2,2), X2(2,3), tol);
param.method = 'backward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[backward 2D dim=2] filled', X(2,4), X2(2,3), tol);

%% nan's at beginning/end stay nan if there's only one impute dim
X = magic(4);
X(1,3) = nan;
X(4,1) = nan;

param.impute_dimension = 1;
param.method = 'forward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 2D dim=1] one Nan remains', [X(3,1) true], [X2(4,1) isnan(X2(1,3))], tol);
param.method = 'backward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[backward 2D dim=1] one Nan remains', [true X(2,3)], [isnan(X2(4,1)) X2(1,3)], tol);

param.impute_dimension = 2;
param.method = 'forward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 2D dim=2] one Nan remains', [true X(1,2)], [isnan(X2(4,1)) X2(1,3)], tol);
param.method = 'backward';
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[backward 2D dim=2] one Nan remains', [X(4,2) X(1,4)], [X2(4,1) X2(1,3)], tol);

%% nan's disappear if there's two ipute dims
X = magic(4);
X(1,3) = nan;
X(3,1) = nan;

param.method = 'forward';
param.impute_dimension = [1, 2];
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 2D dim=1,2] all Nans gone', [X(2,1) X(1,2)], [X2(3,1) X2(1,3)], tol);

%% non-imputed nans should get fill value
clabel = ones(5,1);
X = magic(5);
X(1:3,1:2) = nan;
X(end-2:end,end-1:end) = nan;

param.impute_dimension = 1;
param.method = 'forward';
param.fill = -100;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('non-imputed nans should get fill value', param.fill * ones(3,2), X2(1:3,1:2), tol);

%% order of imputation matters
X = magic(4);
X(2,3) = nan;
X(3,2) = nan;

param.method = 'forward';
param.impute_dimension = [1, 2];
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
param.impute_dimension = [2, 1];
[~, X3, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 2D dim=1,2] order of imputation dims matters', false, all(X2(:)==X3(:)) , tol);

param.method = 'backward';
param.impute_dimension = [1, 2];
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
param.impute_dimension = [2, 1];
[~, X3, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[backward 2D dim=1,2] order of imputation dims matters', false, all(X2(:)==X3(:)) , tol);

%% 3D 
clabel = ones(3,1);
X = round(randn(3,4,5,4)*20);
X(2,2:2:end,2:2:end,:,:)=nan; % leave first row/col without nan to make forward fill work
param.impute_dimension = [1 2 3];

[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);

print_unittest_result('[forward 3D dim=1,2,3] all Nans gone', true, ~any(isnan(X2(:))), tol);

%% 4D
clabel = ones(3,1);
X = round(randn(3,4,5,4)*20);
X(2,3,4,4)= nan;
X(2,2,3,2)= nan;

param.method = 'forward';
param.impute_dimension = 1;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 3D dim=1]', [X(1,3,4,4) X(1,2,3,2)], [X2(2,3,4,4) X2(2,2,3,2)], tol);

param.impute_dimension = 2;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 3D dim=2]', [X(2,2,4,4) X(2,1,3,2)], [X2(2,3,4,4) X2(2,2,3,2)], tol);

param.impute_dimension = 3;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 3D dim=3]', [X(2,3,3,4) X(2,2,2,2)], [X2(2,3,4,4) X2(2,2,3,2)], tol);

param.impute_dimension = 4;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 3D dim=4]', [X(2,3,4,3) X(2,2,3,1)], [X2(2,3,4,4) X2(2,2,3,2)], tol);

%% 4D non-nan values stay the same
clabel = ones(3,1);
X = round(randn(3,4,5,4)*20);
X(2:3:end) = nan;

param.impute_dimension = 2;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward] non-nan values stay the same', X(~isnan(X)), X2(~isnan(X)), tol);

%% 5D: if there's no Nans result is the same
clabel = ones(3,1);
X = round(randn(3,4,5,4,6)*20);

param.impute_dimension = [4 1];
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[forward 5D dim] unchanged with no Nans', X, X2, tol);

%% 5D: nan and inf are dealt with equally
clabel = ones(3,1);
X = round(randn(3,4,5,4,6)*20);
Xnan = X;
Xinf = X;
Xnan(2:3:end) = nan;
Xinf(2:3:end) = inf;

param.method = 'forward';
param.impute_dimension = [4 1];
[~, X2nan, ~] = mv_preprocess_impute_nan(param, Xnan, clabel);
[~, X2inf, ~] = mv_preprocess_impute_nan(param, Xinf, clabel);
print_unittest_result('[forward 5D dim] nan and inf are dealt with equally', X2nan, X2inf, tol);

%% method='random'
clabel = ones(10,1);
clabel(6:10) = 2;
X = magic(10);
X = reshape(X, [10, 5, 2]);
X(1, :, :) = nan;
X(5, 1:3, :) = nan;
X(7, 2, 2) = nan;

param = mv_get_preprocess_param('impute_nan');
param.impute_dimension = 1;
param.method = 'random';
param.use_clabel = 0;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[random, use_clabel=0] all nans gone', true, ~any(isnan(X2(:))), tol);
print_unittest_result('[random, use_clabel=0] non-nans unchanged', X(~isnan(X)), X2(~isnan(X)), tol);

% for class 1 all except row 4 have Nans
X(2,:) = nan;
X(3,1:2:end) = nan;
param.use_clabel = 1;
[~, X2, ~] = mv_preprocess_impute_nan(param, X, clabel);
print_unittest_result('[random, use_clabel=1] all nans imputed by row 4', true, ...
    all(X(4, isnan(X(1,:)))==X2(1, isnan(X(1,:)))) && all(X(4, isnan(X(2,:)))==X2(2, isnan(X(2,:)))) && all(X(4, isnan(X(3,:)))==X2(3, isnan(X(3,:)))) && all(X2(4, isnan(X(5,:)))==X2(5, isnan(X(5,:))))...
    , tol);
