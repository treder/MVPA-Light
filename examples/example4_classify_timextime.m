%%% Time generalisation example
clear all

% Load data (in /examples folder)
[dat, clabel] = load_example_data('epoched3', 0);

%% Setup configuration struct

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 10 repetitions. As classifier, we
% use LDA. The value of the regularisation parameter lambda is determined 
% automatically.
cfg =  [];
cfg.classifier = 'lda';
cfg.normalise  = 'demean';  % 'demean' 'none'
cfg.metric     = 'acc';

[acc, result_acc] = mv_classify_timextime(cfg, dat.trial, clabel);

cfg.metric     = 'auc';
[auc, result_auc] = mv_classify_timextime(cfg, dat.trial, clabel);

%% Plot time generalisation matrix
% close all
% mv_plot_result(result_acc, dat.time, dat.time) % 2nd and 3rd argument are optional

figure
cfg_plot= [];
cfg_plot.x   = dat.time;
cfg_plot.y   = cfg_plot.x;
mv_plot_2D(cfg_plot, acc);
colormap jet
title('Accuracy')

figure
mv_plot_2D(cfg_plot, auc);
colormap jet
title('AUC')

%% Compare with and without cross-validation
cfg.CV      = 'none';
cfg.metric     = 'acc';
[acc_noCV, result_acc_noCV] = mv_classify_timextime(cfg, dat.trial, clabel);

mv_plot_result({result_acc, result_acc_noCV}, dat.time, dat.time)

%% Compare accuracy/AUC when no normalisation is performed
cfg.normalise  = 'none';
cfg.metric     = 'acc';
acc = mv_classify_timextime(cfg, dat.trial, clabel);

cfg.metric     = 'auc';
auc = mv_classify_timextime(cfg, dat.trial, clabel);

figure
mv_plot_2D(cfg_plot, acc);
colormap jet
title('Accuracy')

figure
mv_plot_2D(cfg_plot, auc);
colormap jet
title('AUC')

%% Generalisation with two datasets
% The classifier is trained on one dataset, and tested on another dataset.
% As two datasets, two different subjects are taken. 

[dat, clabel] = load_example_data('epoched3', 0);
% Load data from a different subject (epoched1). This will served as the 
% test data.
% The subject loaded above will serve as training data.
[dat2, clabel2] = load_example_data('epoched1');

cfg =  [];
cfg.classifier = 'lda';
cfg.normalise  = 'zscore';  % 'demean' 'none' 'zscore'
cfg.metric     = 'acc';

[acc31, result31] = mv_classify_timextime(cfg, dat.trial, clabel, dat2.trial, clabel2);

% Reverse the analysis: train the classifier on epoched1, test on epoched3
[acc13, result13]= mv_classify_timextime(cfg, dat2.trial, clabel2, dat.trial, clabel);

% Train AND test on epoched1 (overfitting!)
[acc11, result11]= mv_classify_timextime(cfg, dat2.trial, clabel2, dat2.trial, clabel2);

figure
cfg_plot =[];
cfg_plot.y = dat.time; cfg_plot.x = dat2.time;
mv_plot_2D(cfg_plot, acc31 );
colormap jet
title('Training on epoched3, testing on epoched1')

figure
cfg_plot.x = dat.time; cfg_plot.y = dat2.time;
mv_plot_2D(cfg_plot, acc13 );
colormap jet
title('Training on epoched1, testing on epoched3')

figure
cfg_plot.x = dat.time; cfg_plot.y = dat.time;
mv_plot_2D(cfg_plot, acc11 );
colormap jet
title('Training AND testing on epoched1 (overfitting!)')

% close all
% mv_plot_result({result13, result31}, dat.time, dat.time)
