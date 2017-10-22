%%% Train and test logistic regression.
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

% Average activity in 0.6-0.8 interval (see example1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

X = zscore(X);
%% SVM
close all

param = mv_get_classifier_param('svm');
param.C = 1;

tic
rng(1)
profile on
cf = train_svm(param, X, clabel);
profile viewer
profile off
toc
%%
[predlabel, dval] = test_svm(cf, X);

% Calculate AUC
auc = mv_calculate_performance('auc', dval, clabel);
