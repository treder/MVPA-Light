%%% In example 1, training and testing was performed on the same data. This
%%% can lead to overfitting and an inflated measure of classification
%%% accuracy. The function mv_crossvalidate is used for this purpose.

close all
clear all

% Load data (in /examples folder)
load('epoched3')

% Create class labels (+1's and -1's)
label = zeros(nTrial, 1);
label(attended_deviant)  = 1;   % Class 1: attended deviants
label(~attended_deviant) = -1;  % Class 2: unattended deviants

% Average activity in 0.6-0.8 interval (see example 1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

%% Cross-validation
cfg = [];
cfg.classifier      = 'lda';
cfg.param           = struct('lambda','auto');
cfg.metric          = 'acc';
cfg.CV              = 'kfold';
cfg.K               = 5;
cfg.repeat          = 3;
cfg.balance         = 'undersample';
cfg.verbose         = 1;

acc = mv_crossvalidate(cfg, X, label);

%% Comparing cross-validation to train-test on the same data
% Select only 29 first samples
label29 = label(1:29);
X29 = X(1:29,:);

cfg= [];
cfg.verbose      = 1;
acc = mv_crossvalidate(cfg, X29, label29);

cfg.CV     = 'none';
acc29 = mv_crossvalidate(cfg, X29, label29);

