%%% Train and test a classifier "by hand", i.e. without the
%%% crossvalidation and classification across time functions provided by
%%% MVPA-Light

close all
clear all

% Load data (in /examples folder)
load('epoched3')

% Create class labels (+1's and -1's)
label = zeros(nTrial, 1);
label(attended_deviant)  = 1;   % Class 1: attended deviants
label(~attended_deviant) = -1;  % Class 2: unattended deviants

%% Let's have a look at the data first: Calculate and plot ERP for attended and unattended deviants

% ERP for each condition
erp_attended = squeeze(mean(dat.trial(attended_deviant,:,:)));
erp_unattended = squeeze(mean(dat.trial(~attended_deviant,:,:)));

% Plot ERP: attended deviants in red, unattended deviants in green. Each
% line is one EEG channel.
plot(dat.time, erp_attended, 'r'), hold on
plot(dat.time, erp_unattended, 'g')
grid on

%% Train and test classifier

% Looking at the ERP the classes seem to be well-separated between in the
% interval 0.6-0.8 seconds. We will apply a classifier to this interval. First, 
% find the sample corresponding to this interval, and then average the
% activity across time within this interval. Then use the averaged activity
% for classification.

ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);

X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Get default hyperparameters
param_lda = mv_classifier_defaults('lda');

% Train an LDA classifier
cf_lda = train_lda(X, label, param_lda);

% Test classifier on the same data: the function gives the predicted
% labels (predlabel) and the decision values (dval) which represent the
% distance to the hyperplane
[predlabel, dval] = test_lda(cf_lda, X);

% To calculate classification accuracy, compare the predicted labels to
% the true labels and take the mean
fprintf('Classification accuracy: %2.2f\n', mean(predlabel==label))

% Look at the distribution of the decision values. dval should be positive
% for label +1 (attended deviant) and negative for label -1 (unattended
% deviant)
figure
boxplot(dval, label)