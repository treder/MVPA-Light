%%% In example 1, training and testing was performed on the same data. This
%%% can lead to overfitting and an inflated measure of classification
%%% accuracy. The function mv_crossvalidate is used for this purpose.
close all
clear all

% Load data (in /examples folder)
[dat, clabel, chans] = load_example_data('epoched3');

% Average activity in 0.6-0.8 interval (see example 1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

%% Cross-validation

% Configuration struct for cross-validation. As classifier, we
% use LDA. The value of the regularisation parameter lambda is determined 
% automatically. As performance measure, use area under the ROC curve
% ('auc').
%
% To get a realistic estimate of classification performance, we perform 
% 5-fold (cfg.k = 5) cross-validation with 10 repetitions (cfg.repeat = 10).

cfg_LDA = [];
cfg_LDA.classifier      = 'lda';
cfg_LDA.param           = struct('lambda','auto');
cfg_LDA.metric          = 'auc';
cfg_LDA.cv              = 'kfold';  % 'kfold' 'leaveout' 'holdout'
cfg_LDA.k               = 5;
cfg_LDA.repeat          = 10;
cfg_LDA.balance         = 'undersample';

[acc_LDA, result_LDA] = mv_crossvalidate(cfg_LDA, X, clabel);

% Compare the result for LDA to Logistic Regression (LR).
cfg_LR = cfg_LDA;
cfg_LR.classifier       = 'logreg';
cfg_LR.param            = [];
cfg_LR.param.lambda     = 'auto';

[acc_LR, result_LR] = mv_crossvalidate(cfg_LR, X, clabel);

fprintf('\nClassification accuracy (LDA): %2.2f%%\n', 100*acc_LDA)
fprintf('Classification accuracy (Logreg): %2.2f%%\n', 100*acc_LR)

% Produce plot of results
h = mv_plot_result({result_LDA, result_LR});

%% Use a binomial test to assess statistical significance of accuracies (ACC)
cfg = [];
cfg.test    = 'binomial';

stat = mv_statistics(cfg, result_LDA);

%% Comparing cross-validation to train-test on the same data
cfg_LDA.metric = 'accuracy';

% Select only the first samples
nReduced = 29;
label_reduced = clabel(1:nReduced);
X_reduced = X(1:nReduced,:);

% Cross-validation (proper way)
cfg_LDA.cv = 'kfold';
acc_LDA = mv_crossvalidate(cfg_LDA, X_reduced, label_reduced);

% No cross-validation (test on training data)
cfg_LDA.cv     = 'none';
acc_reduced = mv_crossvalidate(cfg_LDA, X_reduced, label_reduced);

fprintf('Using %d samples with cross-validation (proper way): %2.2f%%\n', nReduced, 100*acc_LDA)
fprintf('Using %d samples without cross-validation (test on training data): %2.2f%%\n', nReduced, 100*acc_reduced)
