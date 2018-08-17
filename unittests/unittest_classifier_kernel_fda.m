% Classifier unit test
%
% Classifier: kernel_fda

rng(42)   %% do not change - might affect the results
tol = 10e-10;
mf = mfilename;

%%% Create Gaussian data
nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

[X_gauss,clabel_gauss] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);


%% check classifier on multi-class spiral data: linear classifier should near chance, RBF kernel should be near 100%

% Create spiral data
N = 1000;
nrevolutions = 2;       % how often each class spins around the zero point
nclasses = 4;
prop = 'equal';
scale = 0;
[X_spiral, clabel_spiral] = simulate_spiral_data(N, nrevolutions, nclasses, prop, scale, 0);

%%% LINEAR kernel: cross-validation
cfg             = [];
cfg.classifier  = 'kernel_fda';
cfg.feedback    = 0;

acc_linear = mv_crossvalidate(cfg,X_spiral,clabel_spiral);

%%% RBF kernel: cross-validation
cfg.param.kernel    = 'rbf';
cfg.param.gamma     = 10e1;
acc_rbf = mv_crossvalidate(cfg,X_spiral,clabel_spiral);

% Since CV is a bit chance-dependent: tolerance of 2%
tol = 0.02;

% For linear kernel: close to chance?
print_unittest_result('classif spiral data (linear kernel)',1/nclasses, acc_linear, tol);

% For RBF kernel: close to 1
print_unittest_result('classif spiral data (RBF kernel)',1, acc_rbf, tol);

%% providing kernel matrix directly VS calculating it from scratch should give same result
gamma = 10e1;

% Get classifier params
param = mv_get_classifier_param('kernel_fda');
param.gamma  = gamma;
param.kernel = 'rbf';

% 1 -provide kernel matrix directly
K = rbf_kernel(struct('gamma',gamma),X_spiral);
param.kernel_matrix = K;
cf_kernel = train_kernel_fda(param, X_spiral, clabel_spiral);

% 2 - do not provide kernel matrix (it is calculated in train_kernel_fda)
param.kernel_matrix = [];
cf_nokernel = train_kernel_fda(param, X_spiral, clabel_spiral);

% Compare solutions - the discriminant
% axes can be in different order, so we look whether there's (nclasses-1)
% 1's in the cross-correlation matrix
C = abs(cf_kernel.A' * cf_nokernel.A); % cross-correlation since all axes have norm = 1
C = sort(C(:),'descend'); % find the largest correlation values
d = all(C(1:nclasses-1) - 1 < 10e-4);

% Are all returned values between 0 and 1?
print_unittest_result('providing kernel matrix vs calculating it from scratch should be equal',1, d, tol);


%% kernel FDA with linear kernel and multiclass LDA should yield the same result
X_gauss = zscore(X_gauss);

%%% FDA
param_fda = mv_get_classifier_param('kernel_fda');
param_fda.kernel = 'linear';
param_fda.reg       = 'ridge';
param_fda.lambda    = 1;
cf_fda = train_kernel_fda(param_fda, X_gauss, clabel_gauss);

w_fda= cf_fda.Xtrain' * cf_fda.A;

%%% Multiclass LDA
param_lda = mv_get_classifier_param('multiclass_lda');
param_lda.reg       = param_fda.reg;
param_lda.lambda    = param_fda.lambda;
cf_lda = train_multiclass_lda(param_lda, X_gauss, clabel_gauss);

corr([w_fda, cf_lda.W])


%%
% Since CV is a bit chance-dependent: tolerance of 2%
tol = 0.02;

% For linear kernel: close to chance?
print_unittest_result('classif spiral data (linear kernel)',1/nclasses, acc_linear, tol);

% For RBF kernel: close to 1
print_unittest_result('classif spiral data (RBF kernel)',1, acc_rbf, tol);

%% todo  -- rest not finished

%% check "lambda" parameter: if lambda = 1, w should be collinear with the difference between the class means
% Get classifier params
param = mv_get_classifier_param('lda');
param.reg       = 'shrink';
param.lambda    = 1;

cf = train_lda(param, X_spiral, clabel_spiral);

% Difference between class means
m = mean(X_spiral(clabel_spiral==1,:)) - mean(X_spiral(clabel_spiral==2,:));

% Correlation between m and cf.w
p = corr(m', cf.w);

% Are all returned values between 0 and 1?
print_unittest_result('check w parameter for lambda=1 (equal to diff of class means?)',1, p,  tol);

%% Cross-validation: performance for well-separated classes should be 100%
nsamples = 100;
nfeatures = 10;
nclasses = 5;
prop = [];
scale = 0.0001;
do_plot = 0;

[X_spiral,clabel_spiral] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

% Plot the data
% close all, plot(X(clabel==1,1),X(clabel==1,2),'.')
% hold all, plot(X(clabel==2,1),X(clabel==2,2),'+')
% figure(1)

expect = 1;

cfg = [];
cfg.feedback        = 0;
cfg.metric          = 'acc';
cfg.classifier      = 'kernel_fda';
cfg.param           = [];
cfg.param.kernel    = 'linear';

actual = mv_crossvalidate(cfg, X_spiral, clabel_spiral);

print_unittest_result('CV for well-separated data',expect, actual, tol);

%% Equivalence between ridge and shrinkage regularisation

% Get classifier param for shrinkage regularisation
param_shrink = mv_get_classifier_param('lda');
param_shrink.reg   = 'shrink';
param_shrink.lambda = 0.5;

% Determine within-class scatter matrix (we need its trace)
Sw= sum(clabel_spiral==1) * cov(X_spiral(clabel_spiral==1,:),1) + sum(clabel_spiral==2) * cov(X_spiral(clabel_spiral==2,:),1);

% Determine the equivalent ridge parameter using the formula
% ridge = shrink/(1-shrink) * trace(C)/P
% Obviously the formula only works for shrink < 1
param_ridge = param_shrink;
param_ridge.reg      = 'ridge';
param_ridge.lambda   = param_shrink.lambda/(1-param_shrink.lambda) * trace(Sw)/nfeatures;

% Train classifiers with both types of regularisation
cf_shrink = train_lda(param_shrink, X_spiral, clabel_spiral);
cf_ridge = train_lda(param_ridge, X_spiral, clabel_spiral);

p = corr(cf_ridge.w, cf_shrink.w);

print_unittest_result('Corr between ridge and shrinkage classifier weights',1, p, tol);
