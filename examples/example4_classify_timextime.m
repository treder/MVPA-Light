%%% Time generalisation example
clear all

% Load data (in /examples folder)
[dat, clabel] = load_example_data('epoched3');

%% Setup configuration struct

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 10 repetitions. As classifier, we
% use LDA. The value of the regularisation parameter lambda is determined 
% automatically.
cfg =  [];
cfg.classifier = 'lda';
cfg.param      = struct('lambda','auto');
cfg.normalise  = 'demean';  % 'demean' 'none'
cfg.metric     = 'acc';

[acc, result_acc] = mv_classify_timextime(cfg, dat.trial, clabel);

cfg.metric     = 'auc';
[auc, result_auc] = mv_classify_timextime(cfg, dat.trial, clabel);

%% Plot time generalisation matrix
close all
mv_plot_result(result_acc, dat.time, dat.time) % 2nd and 3rd argument are optional


figure
cfg= [];
cfg.x   = dat.time;
cfg.y   = cfg.x;
mv_plot_2D(cfg, acc);
colormap jet
title('Accuracy')

figure
cfg= [];
cfg.x   = dat.time;
cfg.y   = cfg.x;
mv_plot_2D(cfg, auc);
colormap jet
title('AUC')

%% Compare accuracy/AUC when no normalisation is performed
cfg.normalise  = 'none';
cfg.metric     = 'acc';
acc = mv_classify_timextime(cfg, dat.trial, clabel);

cfg.metric     = 'auc';
auc = mv_classify_timextime(cfg, dat.trial, clabel);

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
[dat2, clabel2] = load_example_data('epoched1');

cfg =  [];
cfg.classifier = 'lda';
cfg.param      = struct('lambda','auto');
cfg.normalise  = 'demean';  % 'demean' 'none'
cfg.metric     = 'acc';

[acc31, result31] = mv_classify_timextime(cfg, dat.trial, clabel, dat2.trial, clabel2);

% Reverse the analysis: train the classifier on epoched1, test on epoched3
[acc13, result13]= mv_classify_timextime(cfg, dat2.trial, clabel2, dat.trial, clabel);

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

close all
mv_plot_result({result13, result31}, dat.time, dat.time)
