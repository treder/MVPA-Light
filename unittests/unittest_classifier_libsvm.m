% Classifier unit test
%
% Classifier: libsvm

% Note: the classifier itself is not being tested since it's external but
% rather the MVPA-Light interface. 


rng(42)   %% do not change - might affect the results
tol = 10e-10;
mf = mfilename;

% We must perform sanity checks for multiple cases here:
% (1) no svmtrain function available
% (2) an svmtrain function is available, but it is the one by matlab
% (3) two svmtrain function are available (Matlab's one and LIBSVM's one)
%     but the LIBSVM one is overshadowed by Matlab's one
return
%% --- TODO ---

check = which('svmtrain','-all');
if isempty(check)
    warning('LIBSVM is not available or not in the path, skipping unit test')
    return
elseif numel(check)>2
    % Try to find out 
    l = cellfun(@isempty, strfind(check, 'libsvm'))
end

%%% Create Gaussian data
nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

[X_gauss,clabel_gauss] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

% Get classifier params
param = mv_get_classifier_param('libsvm');
param.prob = 1;

% Train and test classifier
cf = train_libsvm(param, X, clabel);
[~,~,prob] = test_logreg(cf,X);

%% test pre-computed kernel

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

% corr([w_fda, cf_lda.W])

% Since CV is a bit chance-dependent: tolerance of 2%
tol = 0.02;

% For linear kernel: close to chance?
print_unittest_result('classif spiral data (linear kernel)',1/nclasses, acc_linear, tol);

% For RBF kernel: close to 1
print_unittest_result('classif spiral data (RBF kernel)',1, acc_rbf, tol);

