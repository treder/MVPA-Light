% Classifier unit test
%
% Classifier: libsvm

% Note: the classifier itself is not being tested since it's external but
% rather the MVPA-Light interface. 


rng(42)   %% do not change - might affect the results
tol = 10e-10;
mf = mfilename;

%% first check whether LIBSVM is available
check = which('svmtrain','-all');
if isempty(check)
    warning('LIBSVM is not available or not in the path, skipping unit test')
    return
else
    try
        % this should work fine with libsvm but crash for Matlab's
        % svmtrain
        svmtrain(0,0,'-q');
    catch
        if numel(check)==1
            % there is an svmtrain but it seems to be Matlab's one
            warning('Found an svmtrain() function but it does not seem to be LIBSVM''s one, skipping unit test')
        else
            % there is multiple svmtrain functions
            warning('Found multiple functions called svmtrain: LIBSVM''s svmtrain() is either not available or overshadowed by another svmtrain function, skipping unit test')
        end
        return
    end
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
param = mv_get_hyperparameter('libsvm');
param.gamma = [0.1, 0.01, 1];

%% use cross-validation [no test, just look for crashes]
param.cv = 10;
cf = train_libsvm(param, X, clabel);

%% check classifier on multi-class spiral data: linear classifier should near chance, RBF kernel should be near 100%

% Create spiral data
N = 1000;
nrevolutions = 2;       % how often each class spins around the zero point
nclasses = 2;
prop = 'equal';
scale = 0;
[X_spiral, clabel_spiral] = simulate_spiral_data(N, nrevolutions, nclasses, prop, scale, 0);

%%% LINEAR kernel: cross-validation
cfg                         = [];
cfg.classifier              = 'libsvm';
cfg.hyperparameter          = [];
cfg.hyperparameter.kernel   = 'linear';
cfg.feedback                = 0;

acc_linear = mv_crossvalidate(cfg, X_spiral, clabel_spiral);

%%% RBF kernel: cross-validation
cfg                         = [];
cfg.classifier              = 'libsvm';
cfg.hyperparameter          = [];
cfg.hyperparameter.kernel   = 'rbf';
cfg.hyperparameter.gamma    = 100;
cfg.feedback                = 0;

acc_rbf = mv_crossvalidate(cfg, X_spiral, clabel_spiral);

% Since CV is a bit chance-dependent: tolerance of 2%
tol = 0.02;

% For linear kernel: close to chance?
print_unittest_result('classif spiral data (linear kernel)',1/nclasses, acc_linear, tol);

% For RBF kernel: close to 1
print_unittest_result('classif spiral data (RBF kernel)',1, acc_rbf, tol);


%% call train/test function with a precomputed kernel
K = rbf_kernel(struct('gamma',500), X_spiral);
[Xtrain, trainlabel, Xtest, testlabel] = mv_select_train_and_test_data(K, clabel, 1:2:numel(clabel), 2:2:numel(clabel), 1);

param = mv_get_hyperparameter('libsvm');
param.kernel = 'precomputed';

cf = train_libsvm(param, Xtrain, trainlabel);

pred_trainlabel = test_libsvm(cf, Xtrain);
pred_testlabel = test_libsvm(cf, Xtest);

fprintf('[precomputed kernel] train accuracy: %2.2f\n', sum(pred_trainlabel==trainlabel)/numel(trainlabel))
fprintf('[precomputed kernel] test accuracy: %2.2f\n', sum(pred_testlabel==testlabel)/numel(testlabel))

%% rbf: providing kernel matrix directly VS calculating it from scratch should give same result
gamma = 10e1;

% Get classifier params
param = mv_get_hyperparameter('libsvm');
param.gamma  = gamma;

% 1 -provide kernel matrix directly
K = rbf_kernel(struct('gamma',gamma),X_spiral);
param.kernel = 'precomputed';
cf_kernel = train_libsvm(param, K, clabel_spiral);

% 2 - do not provide kernel matrix (it is calculated in train_kernel_fda)
param.kernel = 'rbf';
cf_nokernel = train_libsvm(param, X_spiral, clabel_spiral);

% Are all returned values between 0 and 1?
print_unittest_result('[rbf] providing kernel matrix vs calculating it from scratch should be equal',0, norm(cf_kernel.model.sv_coef - cf_nokernel.model.sv_coef), tol);

%% polynomial: providing kernel matrix directly VS calculating it from scratch should give same result
gamma = 10e1;

% Get classifier params
param = mv_get_hyperparameter('libsvm');

% 1 -provide kernel matrix directly
kernel_param = [];
kernel_param.gamma = 1;
kernel_param.coef0 = 1;
kernel_param.degree = 2;

K = polynomial_kernel(kernel_param, X_spiral);
param.kernel = 'precomputed';
cf_kernel = train_libsvm(param, K, clabel_spiral);

% 2 - do not provide kernel matrix (it is calculated in train_kernel_fda)
param.kernel = 'polynomial';
param.degree = kernel_param.degree;
param.coef0 = kernel_param.coef0;
param.gamma = kernel_param.gamma;
cf_nokernel = train_libsvm(param, X_spiral, clabel_spiral);

% Are all returned values between 0 and 1?
print_unittest_result('[polynomial] providing kernel matrix vs calculating it from scratch should be equal',0, norm(cf_kernel.model.sv_coef - cf_nokernel.model.sv_coef), tol);

%% linear: providing kernel matrix directly VS calculating it from scratch should give same result
gamma = 10e1;

% Get classifier params
param = mv_get_hyperparameter('libsvm');

% 1 -provide kernel matrix directly
K = linear_kernel([], X_spiral);
param.kernel = 'precomputed';
cf_kernel = train_libsvm(param, K, clabel_spiral);

% 2 - do not provide kernel matrix (it is calculated in train_kernel_fda)
param.kernel = 'linear';
cf_nokernel = train_libsvm(param, X_spiral, clabel_spiral);

% Are all returned values between 0 and 1?
print_unittest_result('[linear] providing kernel matrix vs calculating it from scratch should be equal',0, norm(cf_kernel.model.sv_coef - cf_nokernel.model.sv_coef), tol);

