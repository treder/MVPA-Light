% Classifier unit test
%
% Classifier: multiclass_lda

rng(42)
tol = 10e-10;
mf = mfilename;


%% for two classes, multiclass_lda should give the same result as lda
X = randn(1000,10);
clabel = randi(2, size(X,1),1);

% Get classifier params for multiclass LDA and binary LDA
param = mv_get_classifier_param('multiclass_lda');
param.reg     = 'ridge';
param.lambda  = 0.1;

param_binary = mv_get_classifier_param('lda');
param_binary.reg     = param.reg;
param_binary.lambda  = param.lambda;

% Train and test classifier
cf_binary = train_lda(param_binary, X, clabel);
cf = train_multiclass_lda(param, X, clabel);

p = corr(cf_binary.w, cf.W);

% Are the weight vectors the same (up to scaling?)
print_unittest_result('correlate weight vectors for multiclass and binary lda',1, abs(p), tol);

%% Cross-validation: performance for well-separated classes should be 100%
nsamples = 120;
nfeatures = 10;
nclasses = 5;
prop = [];
scale = 0.0001;
do_plot = 0;

[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

expect = 1;

cfg = [];
cfg.feedback        = 0;
cfg.metric          = 'acc';
cfg.classifier      = 'multiclass_lda';
cfg.param           = [];
cfg.param.lambda    = 'auto';


actual = mv_crossvalidate(cfg, X, clabel);

print_unittest_result('CV for well-separated data',expect, actual, tol);
