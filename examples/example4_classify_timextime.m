%%% Time generalisation example

clear all

% Load data (in /examples folder)
load('epoched3')
dat.trial = double(dat.trial);

% Create class labels (1's and 2's)
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;   % Class 1: attended deviants
clabel(~attended_deviant) = 2;   % Class 2: unattended deviants

%% Setup configuration struct

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 10 repetitions. As classifier, we
% use LDA. The value of the regularisation parameter lambda is determined 
% automatically.
ccfg =  [];
ccfg.classifier = 'lda';
ccfg.param      = struct('lambda','auto');
ccfg.verbose    = 1;
ccfg.normalise  = 'demean';  % 'demean' 'none'
ccfg.metric     = {'acc' 'auc'};

[acc,auc]= mv_classify_timextime(ccfg, dat.trial, clabel);


%% Plot time generalisation matrix
figure
cfg= [];
cfg.x   = dat.time;
cfg.y   = cfg.x;
mv_plot_2D(cfg, acc);
colormap jet
title('Accuracy')

figure
mv_plot_2D(cfg, auc);
colormap jet
title('AUC')


%% Compare accuracy/AUC when no normalisation is performed
ccfg.normalise  = 'none';
[acc,auc]= mv_classify_timextime(ccfg, dat.trial, clabel);

figure
mv_plot_2D(cfg, acc);
colormap jet
title('Accuracy')

figure
mv_plot_2D(cfg, auc);
colormap jet
title('AUC')

%% Generalisation with two datasets
% The classifier is trained on one dataset, and tested on another dataset.
% As two datasets, two different subjects are taken. 

% Load data from a different subject (epoched1). This will served as the 
% test data.
% The subject loaded above will serve as training data.
dat2 = load('epoched1');
label2 = zeros(dat2.nTrial, 1);
label2(dat2.attended_deviant)  = 1;   % Class 1: attended deviants
label2(~dat2.attended_deviant) = 2;   % Class 2: unattended deviants

dat2 = dat2.dat;

ccfg =  [];
ccfg.classifier = 'lda';
ccfg.param      = struct('lambda','auto');
ccfg.verbose    = 1;
ccfg.normalise  = 'demean';  % 'demean' 'none'
ccfg.metric     = 'acc';

acc31 = mv_classify_timextime(ccfg, dat.trial, clabel, dat2.trial, label2);

% Reverse the analysis: train the classifier on epoched1, test on epoched3
acc13 = mv_classify_timextime(ccfg, dat2.trial, label2, dat.trial, clabel);

figure
cfg =[];
cfg.y = dat.time; cfg.x = dat2.time;
mv_plot_2D(cfg, acc31 );
colormap jet
title('Training on epoched3, testing on epoched1')

figure
cfg.x = dat.time; cfg.y = dat2.time;
mv_plot_2D(cfg, acc13 );
colormap jet
title('Training on epoched1, testing on epoched3')

