%%% In this example we look at classification tasks that involve
%%% multi-dimensional data that cannot be performed with any of the other
%%% high-level functions (mv_searchlight, mv_crossvalidate,
%%% mv_classify_across_time, mv_classify_timextime).
%%%
%%% To this end, we will use the function mv_classify. It is a
%%% generalization of all the other high-level functions. It can work with
%%% data of any dimensionality and size, any order of the dimensions, and
%%% can perform both generalization and searchlight analysis.
%%%
%%% The most obvious application of the function is higher dimensional data
%%% such as time-frequency data which may have the shape [samples x
%%% channels x frequencies x time points]. 
%%%
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched1');
[dat2,clabel2] = load_example_data('epoched2');
[dat3,clabel3] = load_example_data('epoched3');

dat.trial = zscore(dat.trial);
dat2.trial = zscore(dat2.trial);
dat3.trial = zscore(dat3.trial);

%% PART 1: standard classification tasks with mv_classify
% To get an intuition for working with mv_classify, we will perform
% cross-validation, classification across time, and time x time
% generalization using with the respective functions and then compare it to
% the way it can be performed using mv_classify.

%% Cross-validate (cf example2_crossvalidate)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

cfg = [];
cfg.classifier      = 'lda';
cfg.metric          = 'f1';

rng(42);
perf = mv_crossvalidate(cfg, X, clabel);

% Now we will do the same using mv_classify. We need to reset the random
% number generator to get the same result. The only necessary action is to
% indicate that the samples are in dimension 1 and the features in
% dimension 2 (this is also the default but we set them here explicitly
% just for clarity).
cfg.sample_dimension  = 1;
cfg.feature_dimension  = 2;

perf2 = mv_classify(cfg, X, clabel);

fprintf('mv_crossvalidate result: %2.5f\n', perf)
fprintf('mv_classify result: %2.5f\n', perf2)


%% Classify across time 

%% Time-frequency classification
% For this example, we will first calculate the time-frequency
% representation using the spectrogram function. Note that this requires
% the signal processing toolbox.

sz = size(dat.trial);

% Set parameters for spectrogram
win= kaiser(32, 18);
Fs = 1/mean(diff(dat.time));
noverlap = 30;
nfft = 64;

% get number of frequencies and time points
[S,F,T] = spectrogram(squeeze(dat.trial(nn,cc,:)), win, noverlap, nfft, Fs);

freq = dat; 
freq.trial = zeros([sz(1:2), numel(F), numel(T)]);
freq.dimord = 'rpt_chan_freq_time';
freq.time = T + dat.time(1);
freq.freq = F;

for nn=1:sz(1)
    for cc=1:sz(2)
        S = abs(spectrogram(squeeze(dat.trial(nn,cc,:)), win, noverlap, nfft, Fs));
        freq.trial(nn,cc,:,:) = S;
    end
end


%% Baseline correction
pre = find(freq.time < 0);
BL = mean(freq.trial(:,:,pre,:), 3);

%% Generalization across datasets
% use subjects as an extra variable

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
cfg.metric          = 'auc';
cfg.classifier      = 'svm'; % 'kernel_fda'
cfg.param           = [];
cfg.param.kernel    = 'precomputed'; % indicate that the kernel matrix is precomputed
cfg.preprocess      = 'average_kernel';

group_sizes = [1, 5];
res = cell(1, numel(group_sizes));

for ii=1:2
    cfg.preprocess_param = [];
    cfg.preprocess_param.group_size = group_sizes(ii);
    
    % kernel matrix K serves as input
    [~, res{ii}] = mv_classify_across_time(cfg, K, clabel2);
end

mv_plot_result(res, dat.time)
legend(strcat({'Group size: '}, arrayfun(@(c) {num2str(c)}, group_sizes)), ...
    'location', 'southeast')

%% Preprocessing pipelines: concatenating preprocessing steps
% Multiple preprocessing steps can be concatenated by providing cell arrays
% for .preprocess and .preproces_param

% To illustrate this, let's add a nested z-scoring to the sample averaging.
% MVPA-Light carries out the operations in order: First, z-scoring is
% performed on the data. The resultant data is then averaged.

cfg = [];
cfg.metric          = 'acc';
cfg.classifier      = 'lda';
cfg.preprocess      = {'zscore' 'average_samples' 'demean'};

[perf, res] = mv_classify_across_time(cfg, dat2.trial, clabel2);

%% Preprocessing pipelines: adding optional parameters
% building on the previous example, let's now change the group size 
% for the average_samples operation. It is the second operation in the
% preprocessing pipeline, hence we need to make sure that .preprocess_param
% is a cell array and that we set the group size in the second cell:

cfg.preprocess_param = {};
cfg.preprocess_param{2} = [];
cfg.preprocess_param{2}.group_size = 2;

[perf2, res2] = mv_classify_across_time(cfg, dat2.trial, clabel2);

cfg.preprocess_param

mv_plot_result({res, res2}, dat.time)
legend(strcat({'Group size: '}, {'5' '2'}))

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


