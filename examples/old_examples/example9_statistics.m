%%% In the previous examples, various metrics have been calculated for
%%% classification (and regression in example 8) problems. In many
%%% neuroimaging applications, we also want to quantify the statistical
%%% significance of the metric. Is the classification result significantly
%%% better than what one would expect by chance?
%%%
%%% In MVPA-Light, significance can be tested uing the function
%%% mv_statistics. It returns p-values associated with the classification
%%% or regression results and it has methods for correcting for multiple
%%% comparisons. The following tests are available:
%%%
%%% - binomial test: uses a binomial distribution to calculate the
%%%          probability that the found accuracy is due to chance (under
%%%          the null hypothesis that the classifier performs at chance
%%%          level). Note: The binomial test only works with classification
%%%          accuracy as a metric, and there is no way to control for
%%%          multiple comparisons.
%%% - permutation test: a permutation test is a non-parametric test that
%%%          works with any classification or regression metric. An
%%%          empirical null distribution is calculated by randomly
%%%          permuting the class labels (or response vector in regression)
%%%          and repeating the MVPA many times. We can then count how often
%%%          we find our result (or an even more extreme result) by pure
%%%          chance. From this a p-value can be calculated.
%%% - cluster permutation test: a permutation test does not solve the
%%%          multiple comparisons problem ...

clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

%% Run classification across time
% we start by running an ordinary classification across time analysis.
% Since we want to perform a binomial test below, we need to select 
% classification accuracy as a metric
cfg =  [];
cfg.repeat          = 2;
cfg.classifier      = 'lda';
cfg.metric          = 'accuracy';

[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);

mv_plot_result(result, dat.time)

%% Binomial test
cfg = [];
cfg.test    = 'binomial';

stat = mv_statistics(cfg, result);

mv_plot_result(result, dat.time, 'mask', stat.mask)

%% Permutation test
% A permutation test works with any metric in principle. Here, we choose
% AUC.
cfg =  [];
cfg.repeat          = 1;         % set to 1 to speed up the permutation test
cfg.classifier      = 'lda';
cfg.metric          = 'auc';

[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);

cfg = [];
cfg.test    = 'permutation';
cfg.n_permutations = 500;

stat_permutation = mv_statistics(cfg, result, dat.trial, clabel);

mv_plot_result(result, dat.time, 'mask', stat_permutation.mask)

%% Cluster permutation test
% starting from a permutation test, we can obtain a cluster permutation
% test by setting correctm='cluster'
cfg = [];
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.clustercritval  = 0.6;
cfg.n_permutations  = 500;

stat_cluster = mv_statistics(cfg, result, dat.trial, clabel);

mv_plot_result(result, dat.time, 'mask', stat_cluster.mask)
