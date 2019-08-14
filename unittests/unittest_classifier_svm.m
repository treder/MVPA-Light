% Classifier unit test
%
% Classifier: svm

rng(42)   %% do not change - might affect the results
tol = 10e-10;
mf = mfilename;

%% check classifier on multi-class spiral data: linear classifier should near chance, RBF kernel should be near 100%

% Create spiral data
N = 500;
nrevolutions = 2;       % how often each class spins around the zero point
nclasses = 2;
prop = 'equal';
scale = 0.001;
[X,clabel] = simulate_spiral_data(N, nrevolutions, nclasses, prop, scale, 1);

%%% LINEAR kernel: cross-validation
cfg                 = [];
cfg.classifier      = 'svm';
cfg.hyperparameter  = [];
cfg.hyperparameter.kernel    = 'linear';
cfg.hyperparameter.c         = 10e0;
cfg.feedback        = 0;

acc_linear = mv_crossvalidate(cfg,X,clabel);

%%% RBF kernel: cross-validation
cfg.hyperparameter.kernel    = 'rbf';
cfg.hyperparameter.gamma     = 10e1;
acc_rbf = mv_crossvalidate(cfg,X,clabel);

% Since CV is a bit chance-dependent: tolerance of 2%
tol = 0.05;

% For linear kernel: close to chance?
print_unittest_result('classif spiral data (linear kernel)',1/nclasses, acc_linear, tol);

% For RBF kernel: close to 1
print_unittest_result('classif spiral data (RBF kernel)',1, acc_rbf, tol);

%% providing kernel matrix directly VS calculating it from scratch should give same result

% Get classifier params
param = mv_get_hyperparameter('svm');
param.c      = 1;
param.bias   = 0;
param.gamma  = 1;
param.tolerance = 10e-10;
param.regularize_kernel = 0;

% 1 -provide precomputed kernel matrix
K = rbf_kernel(struct('gamma',1),X);
param.kernel = 'precomputed';
cf_kernel = train_svm(param, K, clabel);

% 2 - do not provide kernel matrix (it is calculated in train_kernel_fda)
param.kernel = 'rbf';
cf_nokernel = train_svm(param, X, clabel);

% Are all returned values between 0 and 1?
print_unittest_result('providing kernel matrix vs calculating it from scratch should be equal',0, norm(cf_kernel.alpha - cf_nokernel.alpha), tol);


%% Check probabilities

% Get classifier params
param = mv_get_hyperparameter('svm');
param.prob      = 1;
param.kernel    = 'rbf';

% Train SVM
cf = train_svm(param, X, clabel);

% Test SVM
[~, dval, prob] = test_svm(cf, X);

% Are all returned values between 0 and 1?
print_unittest_result('all probabilities in [0,1]',1, all(prob>=0 | prob<=1), tol);
