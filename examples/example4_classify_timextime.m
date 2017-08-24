%%% Time generalisation example

clear all

% Load data (in /examples folder)
load('epoched3')

% Create class labels (+1's and -1's)
label = zeros(nTrial, 1);
label(attended_deviant)  = 1;   % Class 1: attended deviants
label(~attended_deviant) = -1;  % Class 2: unattended deviants

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

[acc,auc]= mv_classify_timextime(ccfg, dat.trial, label);


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
[acc,auc]= mv_classify_timextime(ccfg, dat.trial, label);

figure
mv_plot_2D(cfg, acc);
colormap jet
title('Accuracy')

figure
mv_plot_2D(cfg, auc);
colormap jet
title('AUC')
