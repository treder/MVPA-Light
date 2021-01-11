%%% In this example we look into preprocessing pipelines. Preprocessing
%%% includes demeaning, z-scoring, PCA, sample averaging, feature
%%% extraction methods such as Common Spatial Patterns (CSP), and any other
%%% approaches that operate on the data prior to classification.
%%%
%%% An important distinction is between what will be referred to as 
%%% 'global' vs 'nested' preprocessing: In 'global' preprocessing, an
%%% operation is applied to the whole dataset (including both training and
%%% test data) at once before classification is done.
%%%
%%% However, in some cases this can lead to
%%% overfitting. Consider CSP, which makes use of the class labels.
%%% Training a CSP filter on the whole dataset means there is information
%%% transfer between test set and train set: The components used in the
%%% train set have been derived by including information from the test set.
%%% 
%%% Nested preprocessing avoids this by obtaining the parameters form an
%%% operation solely from the train data. The parameters (e.g. principal 
%%% components) extracted from the train set are then applied to the test 
%%% set. This assures that no information from the test set went into the 
%%% preprocessing of the train data.
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');
[dat2,clabel2] = load_example_data('epoched3');

%% --- 'Global' preprocessing ---
% in 'global' preprocessing, the full dataset is preprocessed once before
% the classification analysis is started. Here, we will average the
% samples first.

%% Average samples
% see mv_preprocess_average_samples.m for details on sample averaging

% First, we need to obtain the default preprocessing parameters
% using the function mv_get_preprocess_param.
pparam = mv_get_preprocess_param('average_samples');

% Now we call the preprocessing function by hand. All preprocessing
% functions take preprocess_param, X, and clabel as inputs, and they return
% the same set of parameters.
[pparam, dat.trial_av, clabel_av] = mv_preprocess_average_samples(pparam, dat.trial, clabel);

% we can now take the averaged data and new class labels forward and
% perform classification

cfg = [];
cfg.classifier  = 'lda';
perf = mv_classify_across_time(cfg, dat.trial_av, clabel_av);

%% Z-scoring
% A very useful global preprocessing operation is z-scoring, whereby the
% data is scaled and mean-centered. It is recommended in general, but it is
% important for classifiers such as Logistic Regression for numerical
% reasons. Furthermore, z-scoring brings all features on the same footing,
% irrespective of the units they were measured in originally.

pparam = mv_get_preprocess_param('zscore');
[~, dat.trial] = mv_preprocess_zscore(pparam, dat.trial);


%% Undersampling
pparam = mv_get_preprocess_param('undersample');
fprintf('Number of samples in each class before undersampling: %d %d\n', arrayfun(@(c) sum(clabel==c), 1:max(clabel)))
[~, dat.trial, clabel] = mv_preprocess_undersample(pparam, dat.trial, clabel);
fprintf('Number of samples in each class after undersampling: %d %d\n', arrayfun(@(c) sum(clabel==c), 1:max(clabel)))

%% --- Nested preprocessing ---
% Nested preprocessing can be used in conjunction with the high-level
% functions of MVPA-Light such as mv_classify_across_time. The
% preprocessing is applied on train set and test set separately. In some
% cases (such as z-score), parameters (such as mean and std) are
% calculated on the training data and then applied to the test data.

[dat,clabel] = load_example_data('epoched2');

% If we want to oversample the data, this must be done in a nested fashion:
% only the train data is oversampled, this prevents identical samples
% ending up in both train and test data.
% 
% By setting cfg.preprocess = 'oversample', MVPA-Light will call the
% preprocessing function mv_preprocess_oversample internally to oversample
% the train data.

cfg = [];
cfg.metric          = 'f1';
cfg.classifier      = 'logreg';
cfg.preprocess      = 'oversample';

[~, res] = mv_classify_timextime(cfg, dat.trial, clabel);

mv_plot_result(res);


%% Averaging samples/instances
% Here we explore the average_samples preprocessing method. Samples from
% the same class are split into multiple groups and then the dataset is
% replaced by the group means. This reduces the data and at the same time
% increases SNR. 
% To investigate this 
cfg = [];
cfg.metric          = 'acc';
cfg.classifier      = 'lda';
cfg.preprocess      = 'average_samples';

group_sizes = [1, 3, 8];
res = cell(1, numel(group_sizes));

for ii=1:3
    cfg.preprocess_param = [];
    cfg.preprocess_param.group_size = group_sizes(ii);
    
    [~, res{ii}] = mv_classify_across_time(cfg, dat2.trial, clabel2);
end

%% Plot the results
% We make two observations: increasing group size leads to higher classification
% performance. At the same time, the errorbars get larger. The latter is
% because after averaging, the effective number of samples is decreasing,
% leading to a larger variability. This can partly be mitigated by having
% more repeats.
mv_plot_result(res, dat.time)

legend(strcat({'Group size: '}, arrayfun(@(c) {num2str(c)}, group_sizes)), ...
    'location', 'southeast')

%% Kernel averaging for non-linear classification problems
% Kernel averaging is the generalization of sample averaging to non-linear
% kernels. Instead of averaging in input space, the averages are formed in
% Reproducing Kernel Hilbert Space (RKHS). This can be done efficiently
% using the kernel trick. In order to use kernel averaging, the kernels
% need to be precomputed and the kernel matrix provided as input.

% Precompute a kernel matrix for every time point
kparam = [];
kparam.kernel = 'rbf';
kparam.gamma  = 1/30;
kparam.regularize_kernel = 10e-1;
K = compute_kernel_matrix(kparam, dat2.trial);

cfg = [];
cfg.metric                  = 'auc';
cfg.classifier              = 'svm'; % 'kernel_fda'
cfg.hyperparameter          = [];
cfg.hyperparameter.kernel   = 'precomputed'; % indicate that the kernel matrix is precomputed
cfg.repeat                  = 2;
cfg.preprocess              = 'average_kernel';

group_sizes = [1, 5];
res = cell(1, numel(group_sizes));

rng(12)
for ii=1:2
    cfg.preprocess_param = [];
    cfg.preprocess_param.group_size = group_sizes(ii);
    
    % alternative notation for preprocess_param using a cell array:
    % cfg.preprocess_param = {'group_size' group_sizes(ii)};
    
    % kernel matrix K serves as input
    [~, res{ii}] = mv_classify_across_time(cfg, K, clabel2);
end

mv_plot_result(res, dat.time)
legend(strcat({'Group size: '}, arrayfun(@(c) {num2str(c)}, group_sizes)), ...
    'location', 'southeast')

%% Preprocessing pipelines: concatenating preprocessing steps
% Multiple preprocessing steps can be concatenated by providing cell arrays
% for .preprocess and .preprocess_param

% To illustrate this, let's add a nested PCA to the sample averaging.
% MVPA-Light carries out the operations in order: First, PCA is
% performed on the data. The resultant data is then averaged. We compare
% this to only doing the averaging without PCA. 

cfg = [];
cfg.metric          = 'auc';
cfg.classifier      = 'naive_bayes';
cfg.preprocess      = 'average_samples';

[perf_no_pca, res_no_pca] = mv_classify_across_time(cfg, dat2.trial, clabel2);

cfg.preprocess      = {'pca' 'average_samples'};

[perf_pca, res_pca] = mv_classify_across_time(cfg, dat2.trial, clabel2);

% let's name the results: the names will appear in the legend of the
% combined plot
res_no_pca.name = 'average_samples';
res_pca.name = 'PCA + average_samples';

% plot both results together 
mv_plot_result({res_no_pca, res_pca}, dat.time)

% Conclusion: Naive Bayes is significantly improved when using
% PCA. This is probably due to Naive Bayes assuming independence between
% the features: PCA decorrelates the features which is a necessary (albeit
% not sufficient) condition for independence

%% Preprocessing pipelines: adding optional parameters
% building on the previous example, let's now change the group size 
% for the average_samples operation. It is the second operation in the
% preprocessing pipeline, hence we need to make sure that .preprocess_param
% is a cell array and that we set the group size in the second cell:

cfg.preprocess_param = {};
cfg.preprocess_param{2} = [];
cfg.preprocess_param{2}.group_size = 2;

[perf2, res2] = mv_classify_across_time(cfg, dat2.trial, clabel2);

res2.name = 'average samples (group size 2)';
mv_plot_result({res_no_pca, res2}, dat.time)

%% alternative notation for parameters using a cell array
% instead of providing a struct you can also provide a cell array with
% key-value pairs as parameters
cfg.preprocess_param = {};
cfg.preprocess_param{2} = {'group_size'  2};

[perf2, res2] = mv_classify_across_time(cfg, dat2.trial, clabel2);

res2.name = 'average samples (group size 2)';
mv_plot_result({res_no_pca, res2}, dat.time)

%% Preprocessing pipelines with searchlight
% The pipelines work the same way with all high-level functions. Here, we
% try out a pipeline consisting of oversampling -> z-scoring -> averaging
% with mv_searchlight. We do not specify any neighbours, so every feature
% is classified separately. 

% Load data (in /examples folder)
[dat, clabel, chans] = load_example_data('epoched3');

% We want to classify only on the 300-500 ms window
time_idx = find(dat.time >= 0.3  &  dat.time <= 0.5);

cfg = [];
cfg.average     = 1;
cfg.metric      = 'auc';
cfg.preprocess  = {'oversample' 'zscore' 'average_samples'};
cfg.preprocess_param = {};
cfg.preprocess_param{3} = [];

% run for group sizes 1 and 5
cfg.preprocess_param{3}.group_size = 1;
[auc1, result1] = mv_searchlight(cfg, dat.trial(:,:,time_idx), clabel);

cfg.preprocess_param{3}.group_size = 5;
[auc5, result5] = mv_searchlight(cfg, dat.trial(:,:,time_idx), clabel);

result1.name = 'group size 1';
result5.name = 'group size 5';
mv_plot_result({result1 , result5})

%% Plot classification performance as a topography
cfg_plot = [];
cfg_plot.outline = chans.outline;
figure
mv_plot_topography(cfg_plot, auc1, chans.pos);
set(gca,'CLim',[0.4, 0.8])
title('AUC for group size 1')

figure
mv_plot_topography(cfg_plot, auc5, chans.pos);
set(gca,'CLim',[0.4, 0.8])
title('AUC for group size 5')


