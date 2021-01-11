%%% Train and test logistic regression.
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

% Average activity in 0.6-0.8 interval (see example1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

%% Logistic regression

param = mv_get_hyperparameter('logreg');

cf = train_logreg(param, X, clabel);

    
%% -- Logistic regression
close all

param_lr = mv_get_hyperparameter('logreg');
param_lr.lambda = logspace(-6,3,100); % 2
param_lr.plot = 1;
param_lr.tolerance = 1e-6;
param_lr.polyorder = 3;
tic
rng(1)
cf = train_logreg(param_lr, X, clabel);
toc
[predlabel, dval] = test_logreg(cf, X);

% Calculate AUC
auc = mv_calculate_performance('auc', dval, clabel);
