%%% Train and test logistic regression.
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

pparam = mv_get_preprocess_param('zscore');
[~, dat.trial] = mv_preprocess_zscore(pparam, dat.trial);

% Average activity in 0.6-0.8 interval (see example1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));


%% train SVM
close all

param = mv_get_classifier_param('libsvm');

cf = train_libsvm(param, X, clabel);


%% optimise hyperparameter
param = mv_get_classifier_param('svm');
param.kernel = 'polynomial';
% param.kernel = 'rbf';
param.gamma  = [0.001, 0.01, 0.033, 0.1, 100, 1000];
% param.gamma  = 0.00333;
param.degree = [1, 2, 3];
param.c = 'auto';

cf = train_svm(param, X, clabel);

%% Precompute kernel matrix
cfg = [] ;
cfg.kernel = 'rbf';
cfg.gamma = 0.0333;
cfg.regularize_kernel = 10^-10;

K = compute_kernel_matrix(cfg, dat.trial);

%% create averaged kernel matrix
pparam = mv_get_preprocess_param('average_samples');
pparam.is_kernel_matrix = 1;

[~, K_average, clabel_average] = mv_preprocess_average_samples(pparam, K, clabel);

%% classify across time 
cfg = [] ;
cfg.classifier  = 'svm';
cfg.metric      = 'auc';
cfg.param       = [];
cfg.param.c     = 1;

cfg.param.kernel = 'rbf';

[perf, res] = mv_classify_across_time(cfg, dat.trial, clabel);

mv_plot_result(res, dat.time)

%% train SVM using precomputed kernel matrix
param = mv_get_classifier_param('svm');
param.kernel = 'precomputed';

cf = train_svm(param, squeeze(K(:,:,30)), clabel);

%% classify across time using precomputed kernel matrix
cfg = [] ;
cfg.classifier  = 'svm';
cfg.metric      = 'auc';
cfg.param       = [];
cfg.param.c     = 1;

cfg.param.kernel = 'precomputed';

[perf, res] = mv_classify_across_time(cfg, K, clabel);

mv_plot_result(res, dat.time)

%% classify across time using precomputed averaged kernel matrix
cfg = [] ;
cfg.classifier  = 'svm';
cfg.metric      = 'auc';
cfg.param       = [];
cfg.param.c     = 1;
cfg.param.kernel = 'precomputed';

[perf, res_av] = mv_classify_across_time(cfg, K_average, clabel_average);

mv_plot_result({res, res_av}, dat.time)

%% classify across time using precomputed kernel matrix with nested kernel averaging
cfg = [] ;
cfg.classifier  = 'svm';
cfg.metric      = 'auc';
cfg.param       = [];
cfg.param.c     = 1;
cfg.param.kernel = 'precomputed';

cfg.preprocess          = 'average_samples';
cfg.preprocess_param    = [];
cfg.preprocess_param.is_kernel_matrix = 1;

[perf, res_av2] = mv_classify_across_time(cfg, K, clabel);

mv_plot_result({res, res_av, res_av2}, dat.time)
