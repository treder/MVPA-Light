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

X= zscore(X);

%% Cross-validation
ccfg_LDA = [];
ccfg_LDA.classifier      = 'lda';
ccfg_LDA.param           = struct('lambda','auto');
ccfg_LDA.metric          = 'acc';
ccfg_LDA.CV              = 'kfold';
ccfg_LDA.K               = 5;
ccfg_LDA.repeat          = 3;
ccfg_LDA.balance         = 'undersample';
ccfg_LDA.metric          = 'auc';
ccfg_LDA.verbose         = 1;

rng(1);
acc_LDA = mv_crossvalidate(ccfg_LDA, X, clabel);

% Compare the result for LDA to Logistic Regression (LR).
ccfg_LR = ccfg_LDA;
ccfg_LR.classifier = 'logreg';
ccfg_LR.param      = [];
ccfg_LR.param.tolerance = 1e-5;
ccfg_LR.param.zscore = 0;

rng(1)
acc_LR = mv_crossvalidate(ccfg_LR, X, clabel);

fprintf('\nClassification accuracy (LDA): %2.2f%%\n', 100*acc_LDA)
fprintf('Classification accuracy (Logreg): %2.2f%%\n', 100*acc_LR)

%% Comparing outlier resistance of classifiers
% Create outlier
X_with_outlier = X;
X_with_outlier(10,:) = X_with_outlier(10,:)*100;
X_with_outlier(20,:) = X_with_outlier(20,:)*1000;

rng(1)
acc_LDA = mv_crossvalidate(ccfg_LDA, X_with_outlier, clabel);
rng(1)
acc_LR = mv_crossvalidate(ccfg_LR, X_with_outlier, clabel);

fprintf('\nClassification accuracy (LDA): %2.2f%%\n', 100*acc_LDA)
fprintf('Classification accuracy (Logreg): %2.2f%%\n', 100*acc_LR)

%% Comparing cross-validation to train-test on the same data
% Select only the first samples
nReduced = 29;
label_reduced = clabel(1:nReduced);
X_reduced = X(1:nReduced,:);

ccfg= [];
ccfg.verbose      = 1;
acc_LDA = mv_crossvalidate(ccfg, X_reduced, label_reduced);

ccfg.CV     = 'none';
acc_reduced = mv_crossvalidate(ccfg, X_reduced, label_reduced);

fprintf('Performance using %d samples with cross-validation: %2.2f%%\n', nReduced, 100*acc_LDA)
fprintf('Performance using %d samples without cross-validation (overfitting the training data): %2.2f%%\n', nReduced, 100*acc_reduced)
