% Play around with the SVM implementation
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched3');

ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Get default hyperparameters
param = mv_classifier_defaults('lda');

% Train an LDA classifier
tic
cf = train_lda(param, X, clabel);
toc

%% -- SVM
param = mv_classifier_defaults('svm');
param.lambda = logspace(-5,2,20); % 1
param.plot = 0;
param.tolerance = 1e-6;
param.polyorder=2;
param.predict_regularisation_path = 0;

% param.z1 = 0.5;
% param = mv_classifier_defaults('svm',param);

tic
rng(1);
cf = train_svm(param, zscore(X), clabel);
toc
%%

[predlabel, dval] = test_svm(cf, X);

% Calculate AUC
auc = mv_classifier_performance('auc', dval, clabel);

%% -- Logistic regression
param = mv_classifier_defaults('logreg');
param.lambda = logspace(-6,3,100); % 2
param.plot = 0;
param.tolerance = 1e-6;

tic
cf = train_logreg(param, X, clabel);
% cf = train_logreg(param, X(:,[1,21]), clabel);
toc
%%
[predlabel, dval] = test_logreg(cf, X);

% Calculate AUC
auc = mv_classifier_performance('auc', dval, clabel);

%%
fprintf('Logreg =\t%2.5f sec\nLDA =\t\t%2.5f sec\n',t1,t2)
profile on
for ii=1:100
    cf_lr = train_logreg(param, Xz, clabel);
end

