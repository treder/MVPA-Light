%%% In example 1, training and testing was performed on the same data. This
%%% can lead to overfitting and an inflated measure of classification
%%% accuracy. The function mv_crossvalidate is used for this purpose.

close all
clear all

% Load data (in /examples folder)
load('epoched3')

% Create class labels (1's and 2's)
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;   % Class 1: attended deviants
clabel(~attended_deviant) = 2;   % Class 2: unattended deviants

% Average activity in 0.6-0.8 interval (see example 1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

%% Cross-validation
ccfg = [];
ccfg.classifier      = 'lda';
ccfg.param           = struct('lambda','auto');
ccfg.metric          = 'acc';
ccfg.CV              = 'kfold';
ccfg.K               = 5;
ccfg.repeat          = 3;
ccfg.balance         = 'undersample';
ccfg.metric          = 'auc';
ccfg.verbose         = 1;

acc = mv_crossvalidate(ccfg, X, clabel);

fprintf('\nClassification accuracy: %2.2f%%\n', 100*acc)

%% Comparing cross-validation to train-test on the same data
% Select only the first samples
nReduced = 29;
label_reduced = clabel(1:nReduced);
X_reduced = X(1:nReduced,:);

ccfg= [];
ccfg.verbose      = 1;
acc = mv_crossvalidate(ccfg, X_reduced, label_reduced);

ccfg.CV     = 'none';
acc_reduced = mv_crossvalidate(ccfg, X_reduced, label_reduced);

fprintf('Performance using %d samples with cross-validation: %2.2f%%\n', nReduced, 100*acc)
fprintf('Performance using %d samples without cross-validation (overfitting): %2.2f%%\n', nReduced, 100*acc_reduced)