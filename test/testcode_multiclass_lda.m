% Play around with the multiclass LDA implementation
close all
clear all

% Load Fisher's original iris dataset
load fisheriris

clabel = cellfun(@(x) find(ismember({'setosa','versicolor','virginica'},x)), species);

% Get default hyperparameters
param = mv_get_classifier_param('multiclass_lda');

% Train an multiclass LDA classifier
cf = train_multiclass_lda(param, meas, clabel);

% Test multiclass LDA
predlabel = test_multiclass_lda(cf, meas);
acc = mv_calculate_performance('confusion', predlabel, clabel);

fprintf('Performance on training data: %2.2f\n', acc)

%% Cross-validation

% Accuracy
cfg = [];
cfg.classifier = 'multiclass_lda';
cfg.metric = 'acc';
perf = mv_crossvalidate(cfg, meas, clabel);

% Confusion matrix
cfg = [];
cfg.classifier = 'multiclass_lda';
cfg.metric = 'confusion';

confusion = mv_crossvalidate(cfg, meas, clabel);

