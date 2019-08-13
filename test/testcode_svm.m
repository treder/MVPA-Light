% Play around with the SVM implementation
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched3');

ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Get default hyperparameters
param = mv_get_classifier_param('lda');

% Train an LDA classifier
tic
cf = train_lda(param, X, clabel);
toc

[predlabel, dval] = test_lda(cf, X);
acc = mv_calculate_performance('acc', dval, clabel)


%% -- SVM
param = mv_get_classifier_param('svm');
param.c = logspace(-5,2,10);
param.plot = 0;
% param.k = 5;
param.bias = 0;

% param.gamma = 1;

param.kernel = 'rbf';
% param.kernel = 'polynomial';
% param.kernel = 'linear';

tic
% rng(1);
cf = train_svm(param, zscore(X), clabel);
toc

[predlabel, dval] = test_svm(cf, X);

% Calculate ACC
acc = mv_calculate_performance('acc', dval, clabel)


%% SVM -- crossvalidate
cfg =[] ;
cfg.metric          = 'acc';
cfg.classifier      = 'svm';
cfg.hyperparameter  = [];
cfg.hyperparameter.kernel    = 'rbf';
% cfg.hyperparameter.kernel    = 'polynomial';
% cfg.hyperparameter.kernel    = 'linear';
cfg.hyperparameter.c = 11;

tic
acc = mv_crossvalidate(cfg, X, clabel)
toc

%% LIBSVM
param = mv_get_classifier_param('libsvm');

tic
rng(1);
cf = train_libsvm(param, X, clabel);
toc

[predlabel, dval] = test_libsvm(cf, X);
% auc_libsvm= mv_calculate_performance('auc', dval, clabel)
acc_libsvm= mv_calculate_performance('acc', predlabel, clabel)


%% LIBSVM -- crossvalidate

cfg =[] ;
cfg.metric          = 'acc';
cfg.classifier      = 'libsvm';
cfg.hyperparameter  = [];

cfg.hyperparameter.kernel_type = 0; % linear
cfg.hyperparameter.kernel_type = 1; % polynomial
cfg.hyperparameter.kernel_type = 2; % RBF
% cfg.hyperparameter.kernel_type = 3; % sigmoid

tic
acc = mv_crossvalidate(cfg, X, clabel);
toc

%% -- Logistic regression
param = mv_classifier_defaults('logreg');
param.lambda = logspace(-6,3,100); % 2
param.plot = 0;
param.tolerance = 1e-6;

tic
cf = train_logreg(param, X, clabel);
% cf = train_logreg(param, X(:,[1,21]), clabel);
toc
%%
[predlabel, dval] = test_logreg(cf, X);

% Calculate AUC
auc = mv_classifier_performance('auc', dval, clabel);
