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


%% train kernel FDA
close all

param = mv_get_classifier_param('kernel_fda');
% param.c = 10;
param.tolerance = 10e-10;

tic
rng(1)
% profile on
cf = train_kernel_fda(param, X, clabel);
% profile viewer
% profile off
toc

%% optimise hyperparameter --- not implemented yet
param = mv_get_classifier_param('kernel_fda');
param.kernel = 'polynomial';
% param.kernel = 'rbf';
param.gamma  = [0.001, 0.01, 0.033, 0.1, 100, 1000];
% param.gamma  = 0.00333;
param.degree = [1, 2, 3];
param.c = 'auto';

cf = train_kernel_fda(param, X, clabel);

%% Precompute kernel matrix
cfg = [] ;
cfg.kernel = 'rbf';
cfg.gamma = 0.0333;
cfg.regularize_kernel = 0;

K = compute_kernel_matrix(cfg, dat.trial);

%% create averaged kernel matrix
pparam = mv_get_preprocess_param('average_samples');
pparam.is_kernel_matrix = 1;

[~, K_average, clabel_average] = mv_preprocess_average_samples(pparam, K, clabel);

%% train kernel FDA using precomputed kernel matrix
param = mv_get_classifier_param('kernel_fda');
param.kernel = 'precomputed';

cf = train_kernel_fda(param, squeeze(K(:,:,30)), clabel);

%% classify across time 
cfg = [] ;
cfg.classifier  = 'kernel_fda';
cfg.metric      = 'acc';
cfg.hyperparameter       = [];

cfg.hyperparameter.kernel = 'rbf';
cfg.hyperparameter.gamma  = 0.0333;

tic
[perf, res] = mv_classify_across_time(cfg, dat.trial, clabel);
toc

%% classify across time using precomputed kernel matrix
cfg = [] ;
cfg.classifier  = 'kernel_fda';
cfg.metric      = 'acc';

cfg.hyperparameter       = [];
cfg.hyperparameter.kernel = 'precomputed';

tic
[perf, res2] = mv_classify_across_time(cfg, K, clabel);
toc

mv_plot_result({res, res2}, dat.time)

%% classify across time using precomputed averaged kernel matrix
cfg = [] ;
cfg.classifier  = 'kernel_fda';
cfg.metric      = 'acc';

cfg.hyperparameter       = [];
cfg.hyperparameter.kernel = 'precomputed';

[perf, res_av] = mv_classify_across_time(cfg, K_average, clabel_average);

mv_plot_result({res, res_av}, dat.time)

%% classify across time using precomputed kernel matrix with nested kernel averaging
cfg = [] ;
cfg.classifier  = 'kernel_fda';
cfg.metric      = 'acc';

cfg.hyperparameter       = [];
cfg.hyperparameter.kernel = 'precomputed';

cfg.preprocess          = 'average_samples';
cfg.preprocess_param    = [];
cfg.preprocess_param.is_kernel_matrix = 1;

[perf, res_av2] = mv_classify_across_time(cfg, K, clabel);

mv_plot_result({res, res_av, res_av2}, dat.time)
