%%% Time generalisation example using the function mv_classify_timextime.
%%% In this function, we need data with a time dimension [samples x
%%% features x time points]. Then, cross-validation is run separately for
%%% every combination of training time point and test time point. The
%%% result is a matrix of classification performance scores, one score for
%%% every combination of training and test times.
clear all
close all

% Load data (in /examples folder)
[dat, clabel] = load_example_data('epoched3', 0);

%% Setup configuration struct

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 10 repetitions. As classifier, we
% use LDA.
cfg =  [];
cfg.classifier = 'lda';
cfg.metric     = 'accuracy';
cfg.preprocess = 'zscore';

[acc, result_acc] = mv_classify_timextime(cfg, dat.trial, clabel);

% Let us re-run the classification, this calculating the area the ROC curve
% (AUC) as a performance metric
cfg.metric     = 'auc';
[auc, result_auc] = mv_classify_timextime(cfg, dat.trial, clabel);

%% Plot time generalisation matrix
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

% We already calculated cross-validated performance above. Here, we do the
% analysis once again, this time without cross-validation.
cfg.cv      = 'none';
cfg.metric     = 'accuracy';
[acc_noCV, result_acc_noCV] = mv_classify_timextime(cfg, dat.trial, clabel);

mv_plot_result(result_acc, dat.time, dat.time)
mv_plot_result(result_acc_noCV, dat.time, dat.time)

%% Compare accuracy/AUC when no normalization is performed
% the lack of normalization affects accuracy but not AUC
cfg.preprocess = {};
cfg.metric     = 'accuracy';
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
%
% Note that in this case, nested z-scoring does not make sense since train
% and test set are not directly comparable in terms of their amplitudes.
% Hence, each dataset should be z-scored separately.
% Therefore, we will preprocess both datasets globally prior to classification 
% (see example7 for more information on preprocessing).


[dat, clabel] = load_example_data('epoched3', 0);
% Load data from a different subject (epoched1). This will serve as the 
% test data.
% The subject loaded above will serve as training data.
[dat2, clabel2] = load_example_data('epoched1');

% 'Global' z-scoring prior to classification
preprocess_param = mv_get_preprocess_param('zscore');
[~, dat.trial] = mv_preprocess_zscore(preprocess_param, dat.trial);
[~, dat2.trial] = mv_preprocess_zscore(preprocess_param, dat2.trial);

cfg =  [];
cfg.classifier = 'lda';
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
