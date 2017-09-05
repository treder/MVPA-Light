%%% Train and test a classifier "by hand", i.e. without the
%%% crossvalidation and classification across time functions provided by
%%% MVPA-Light

close all
clear all

% Load data (in /examples folder)
load('epoched3')

% attenden_deviant contains the information about the trials. Use this to
% create the true class labels, indicating whether the trial corresponds to
% an attended deviant (1) or an unattended deviant (2).
truelabel = zeros(nTrial, 1);
truelabel(attended_deviant)  = 1;  % Class 1: attended deviants
truelabel(~attended_deviant) = 2;  % Class 2: unattended deviants

%% Let's have a look at the data first: Calculate and plot ERP for attended and unattended deviants

% ERP for each condition
erp_attended = squeeze(mean(dat.trial(attended_deviant,:,:)));
erp_unattended = squeeze(mean(dat.trial(~attended_deviant,:,:)));

% Plot ERP: attended deviants in red, unattended deviants in green. Each
% line is one EEG channel.
close
plot(dat.time, erp_attended, 'r'), hold on
plot(dat.time, erp_unattended, 'b')
grid on

%% Train and test classifier

% Looking at the ERP the classes seem to be well-separated between in the
% interval 0.6-0.8 seconds. We will apply a classifier to this interval. First, 
% find the sample corresponding to this interval, and then average the
% activity across time within this interval. Then use the averaged activity
% for classification.
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Get default hyperparameters
param = mv_classifier_defaults('lda');

% Train an LDA classifier
cf = train_lda(param, X, truelabel);

% Test classifier on the same data: the function gives the predicted
% labels (predlabel) and the decision values (dval) which represent the
% distance to the hyperplane
[predlabel, dval] = test_lda(cf, X);

% To calculate classification accuracy, compare the predicted labels to
% the true labels and take the mean
fprintf('Classification accuracy: %2.2f\n', mean(predlabel==truelabel))

% Look at the distribution of the decision values. dval should be positive
% for label 1 (attended deviant) and negative for label 2 (unattended
% deviant)
figure
boxplot(dval, truelabel)

%% -- Logistic regression

param = mv_classifier_defaults('logreg');

cf = train_logreg(X, truelabel, param);

%% ---
param = mv_classifier_defaults('logist');
param.eigvalratio = 10^-10;
param.lambda = 10^10;
cf = train_logist(X, truelabel, param);

[predlabel, dval] = test_lda(cf, X);