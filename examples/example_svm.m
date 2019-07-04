%%% Train and test logistic regression.
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

% Average activity in 0.6-0.8 interval (see example1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

X = zscore(X);

%% Precompute kernel matrix
cfg = [] ;
cfg.kernel = 'rbf';
cfg.gamma = 1;
cfg.regularize_kernel = 0;

K = compute_kernel_matrix(cfg, dat.trial);

%% train SVM
close all

param = mv_get_classifier_param('svm');
param.c = 10;
param.tolerance = 10e-10;

tic
rng(1)
% profile on
cf = train_svm(param, X, clabel);
% profile viewer
% profile off
toc

%% train SVM using kernel matrix
param = mv_get_classifier_param('svm');
param.kernel = 'precomputed';

cf = train_svm(param, squeeze(K(:,:,30)), clabel);


%%
[predlabel, dval] = test_svm(cf, X);

% Calculate AUC
auc = mv_calculate_performance('auc', dval, clabel);
