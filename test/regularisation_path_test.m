%%% Train and test a classifier "by hand", i.e. without the
%%% crossvalidation and classification across time functions provided by
%%% MVPA-Light

% Before running the code, cd into the examples subfolder or add it to your
% path temporally

close all
clear all

% Load data (in /examples folder)
load('epoched3')
dat.trial = double(dat.trial);

% attenden_deviant contains the information about the trials. Use this to
% create the true class labels, indicating whether the trial corresponds to
% an attended deviant (1) or an unattended deviant (2).
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;  % Class 1: attended deviants
clabel(~attended_deviant) = 2;  % Class 2: unattended deviants

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


param.lambda = logspace(-6,0,100);
param.plot = 1;

% Train an LDA classifier
cf = train_lda(param, X, clabel);

% Test classifier on the same data: the function gives the predicted
% labels (predlabel) and the decision values (dval) which represent the
% distance to the hyperplane
[predlabel, dval] = test_lda(cf, X);

% To calculate classification accuracy, compare the predicted labels to
% the true labels and take the mean
fprintf('Classification accuracy: %2.2f\n', mean(predlabel==clabel))

% Calculate AUC
auc = mv_classifier_performance('auc', dval, clabel);

% Look at the distribution of the decision values. dvals should be positive
% for clabel 1 (attended deviant) and negative for clabel 2 (unattended
% deviant). dval = 0 is the decision boundary
figure
boxplot(dval, clabel)
hold on
plot(xlim, [0 0],'k--')
ylabel('Decision values')
xlabel('Class')

%% -- Logistic regression
param = mv_classifier_defaults('logreg');
param.lambda = logspace(-6,3,100); % 2
param.plot = 1;
param.tolerance = 1e-6;

tic
cf = train_logreg(param, X, clabel);
% cf = train_logreg(param, X(:,[1,21]), clabel);
toc
[predlabel, dval] = test_logreg(cf, X);

% Calculate AUC
auc = mv_classifier_performance('auc', dval, clabel);

%%
fprintf('Logreg =\t%2.5f sec\nLDA =\t\t%2.5f sec\n',t1,t2)
profile on
for ii=1:100
    cf_lr = train_logreg(param, Xz, clabel);
end

