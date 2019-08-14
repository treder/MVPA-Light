% Play around with the LIBSVM and LIBLINEAR
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched3');

ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Get default hyperparameters
param = mv_get_hyperparameter('lda');

% Train an LDA classifier
tic
cf = train_lda(param, X, clabel);
toc

[predlabel, dval] = test_lda(cf, X);

% Calculate AUC
auc_lda= mv_calculate_performance('auc', dval, clabel)

%% LIBSVM
param = mv_get_hyperparameter('libsvm');

tic
rng(1);
cf = train_libsvm(param, X, clabel);
toc

[predlabel, dval] = test_libsvm(cf, X);
auc_libsvm= mv_calculate_performance('auc', dval, clabel)
acc_libsvm= mv_calculate_performance('acc', predlabel, clabel)

%% LIBLINEAR (log reg)
param = mv_get_hyperparameter('liblinear');
param.type = 0;   % L2-regularised logistic regression
param.C = 1;
param.quiet = 1;

tic
rng(1);
cf = train_liblinear(param, X, clabel);
toc

[predlabel, dval] = test_liblinear(cf, X);
auc_liblinear = mv_calculate_performance('auc', dval, clabel)
acc_liblinear = mv_calculate_performance('acc', predlabel, clabel)

%% LIBLINEAR
param = mv_classifier_defaults('liblinear');
param.type = 3;   % L2-regularised L1-loss SVM
% param.C = 1;
param.cost = 100.1;
param.quiet = 1;

tic
% rng(1);
cf = train_liblinear(param, X, clabel);
toc

[predlabel, dval] = test_liblinear(cf, X);
acc= mv_classifier_performance('acc', predlabel, clabel)

%%

[predlabel, dval] = test_svm(cf, X);

% Calculate AUC
auc = mv_classifier_performance('auc', dval, clabel);
