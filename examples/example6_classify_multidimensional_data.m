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
[dat,clabel,chans] = load_example_data('epoched1');
[dat2,clabel2] = load_example_data('epoched2');
[dat3,clabel3] = load_example_data('epoched3');

dat.trial = zscore(dat.trial);
dat2.trial = zscore(dat2.trial);
dat3.trial = zscore(dat3.trial);

%% PART 1: standard classification tasks with mv_classify
% To get an intuition for working with mv_classify, we will perform
% cross-validation, classification across time, and time x time
% generalization with mv_classify and we will compare it to the way its
% done using the other high-level functions mv_crossvalidate,
% mv_classify_across_time, and mv_classify_timextime.

%% compare to mv_crossvalidate (based on example2_crossvalidate)
% Take average in a time window so that X is [samples x features]
ival_idx = (dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Let's start doing cross-validation with mv_crossvalidate
cfg = [];
cfg.classifier      = 'lda';
cfg.metric          = 'f1';

rng(42);
perf = mv_crossvalidate(cfg, X, clabel);

% Now we will do the same using mv_classify.  We need to indicate 
% that the samples are in dimension 1 and the features in
% dimension 2 (this is also the default but we set them here explicitly
% just for clarity).
cfg.sample_dimension  = 1;
cfg.feature_dimension  = 2;

% optionally, we can also provide the names of the dimensions to get nice output
cfg.dimension_names = {'samples','channels'};

% We also need to reset the random number generator to get the same result.
rng(42);
perf2 = mv_classify(cfg, X, clabel);

fprintf('mv_crossvalidate result: %2.5f\n', perf)
fprintf('mv_classify result: %2.5f\n', perf2)

%% compare to mv_classify_across_time 
cfg =  [];
cfg.classifier      = 'lda';
cfg.metric          = 'precision';

rng(2) % reset random number generator such that random folds are identical
perf = mv_classify_across_time(cfg, dat.trial, clabel);

% To do the same with mv_classify, we have to define the three dimensions.
% Like in the previous example, samples are in dimension 1 and features in
% dimension 2. The last dimension will be automatically devised as a search
% dimension, we do not need to explicitly define it.
cfg.sample_dimension  = 1;
cfg.feature_dimension  = 2;

% optional: provide the names of the dimensions for nice output
cfg.dimension_names = {'samples','channels','time points'};

rng(2) % reset random number generator such that random folds are identical
perf2 = mv_classify(cfg, dat.trial, clabel);

% the results are identical again
figure
plot(perf)
hold all
plot(perf2)

fprintf('Difference between perf and perf2: %2.2f\n', norm(perf-perf2)) % should be 0

%% compare to mv_classify_timextime
cfg =  [];
cfg.classifier      = 'lda';
cfg.metric          = 'auc';

rng(42) % reset random number generator such that random folds are identical
perf = mv_classify_timextime(cfg, dat.trial, clabel);

% Like in the previous example, samples are in dimension 1 and features in
% dimension 2. The last dimension will be automatically devised as search
% dimension. However, this time we also need to indicate that we want 
% the 3rd dimension for generalization (time x time).
cfg.sample_dimension  = 1;
cfg.feature_dimension  = 2;
cfg.generalization_dimension = 3;

% optional: provide the names of the dimensions for nice output
cfg.dimension_names = {'samples','channels','time points'};

rng(42) % reset random number generator such that random folds are identical
perf2 = mv_classify(cfg, dat.trial, clabel);

% the results are identical again
figure
subplot(1,2,1),imagesc(perf)
subplot(1,2,2),imagesc(perf2)

fprintf('Difference between perf and perf2: %2.2f\n', norm(perf-perf2)) % should be 0

%% compare to mv_searchlight

% Average activity in 0.6-0.8 interval (see example 1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Distance matrix giving the pair-wise distances between electrodes
nb = squareform(pdist(chans.pos));

% Turn into binary matrix of 0's and 1's by defining a threshold distance
nb = nb < 0.2;

% Numer of neighbours for each channel
sum(nb)

% using mv_searchlight
cfg = [];
cfg.metric      = 'auc';
cfg.neighbours  = nb;

rng(42)
auc = mv_searchlight(cfg, X, clabel);

% using mv_classify
% we just need to define the sample dimension 1, then dimension 2 will
% automatically become the search dimension. Other than that, the syntax is
% identical
cfg = [];
cfg.metric              = 'auc';
cfg.neighbours          = nb;
cfg.sample_dimension    = 1;
cfg.dimension_names = {'samples','channels'};  % for nice output


rng(42) % reset random number generator such that random folds are identical
auc2 = mv_classify(cfg, X, clabel);


% the results are identical again
figure
bar([auc, auc2])
legend({'auc', 'auc2'})
xlabel('Channel'), ylabel('AUC')
grid on

fprintf('Difference between auc and auc2: %2.2f\n', norm(auc-auc2)) % should be 0


%% PART 2: beyond standard examples

%% Classification across time including a searchlight
% Here we will perform classification across time, like implemented in the
% function mv_classify_across_time. Additionally, we will include a
% searchlight across time that includes neighbouring time points. As a
% reference, let's first run it without the searchlight.

cfg =  [];
cfg.classifier      = 'lda';
cfg.metric          = 'auc';
cfg.sample_dimension  = 1;
cfg.feature_dimension  = 2;

% optional: provide the names of the dimensions for nice output
cfg.dimension_names = {'samples','channels','time points'};

% Let's first just classify without neighbours
[perf, result] = mv_classify(cfg, dat.trial, clabel);

% The third dimension (time) serves as search dimension. Now, for
% every time point, we also want to take up the 2 immediately preceding and
% 2 immediately following time points as features. To this end, we define a
% [time x time] neighbourhood matrix that has ones on the diagonals and
% subdiagonals. This defines the extent of the searchlight.
O = ones(size(dat.trial,3));
O = O - triu(O,3) - tril(O,-3);

O(1:10,1:20)

cfg.neighbours = O;

[perf2, result2] = mv_classify(cfg, dat.trial, clabel);

% the results are identical again
% figure
% plot(perf)
% hold all
% plot(perf2)
% legend({'without searchlight' ,'with searchlight'})

% let's give the results names before combining them and then plot
result.name = 'without searchlight';
result2.name = 'with searchlight';
mv_plot_result({result, result2}, dat.time)

%% --- Time-frequency classification [requires signal processing toolbox] ---
% For this example, we will first calculate the time-frequency
% representation using the spectrogram function. Note that this requires
% the signal processing toolbox.

%% Calculate spectrogram
sz = size(dat.trial);

% Set parameters for spectrogram
win= chebwin(20);
Fs = 1/mean(diff(dat.time));
noverlap = 18;
nfft = 64;

% get number of frequencies and time points
[S,F,T] = spectrogram(squeeze(dat.trial(1,1,:)), win, noverlap, nfft, Fs);

% Correct time T
T = T + dat.time(1);

freq = dat; 
freq.trial = zeros([sz(1:2), numel(F), numel(T)]);
freq.dimord = 'rpt_chan_freq_time';
freq.time = T;
freq.freq = F;

for nn=1:sz(1)
    for cc=1:sz(2)
        S = abs(spectrogram(squeeze(dat.trial(nn,cc,:)), win, noverlap, nfft, Fs));
        freq.trial(nn,cc,:,:) = S;
    end
end

%% Baseline correction
pre = find(freq.time < 0);
BL = mean(mean(freq.trial(:,:,:,pre), 4),1);

% calculate relative baseline 
sz = size(freq.trial);
BLmat = repmat(BL, [sz(1) 1 1 sz(4)]);
freq.trial = (freq.trial - BLmat) ./ BLmat;

%% Classification for each time-frequency point separately
cfg = [];
cfg.classifier      = 'lda';
cfg.metric          = 'acc';

% samples are in dimension 1 and features in
% dimension 2. The last two dimensions  will be automatically devised as search
% dimensions.
cfg.sample_dimension = 1;
cfg.feature_dimension  = 2;

% optional: provide the names of the dimensions for nice output
cfg.dimension_names = {'samples','channels','frequencies', 'time points'};

[perf, result] = mv_classify(cfg, freq.trial, clabel);

% perf is now a 2-D [frequencies x times] matrix of classification results
% figure
% imagesc(T,F,perf)
% colorbar
% xlabel('Time'), ylabel('Frequency')
% title('AUC at each time-frequency point')

% call mv_plot_result: results is the first argument, followed by the
% arguments definining the x-axis (time) and y-axis (frequency)
mv_plot_result(result, freq.time, freq.freq)

%% Classification for each time-frequency point separately including neighbours
% Let's repeat the previous analysis but, for each time-frequency point,
% let's include neighbouring frequencies and time points
% frequencies. To this end, we need to create neighbourhood matrices for
% each of the two search dimensions:

[nsamples, nchannels, nfrequencies, ntimes] = size(freq.trial);

% 1) create binary neighbourhood matrix for frequencies (includes a given
% frequency and the two immediately preceding and following frequencies)
O = ones(nfrequencies);
O = O - triu(O,2) - tril(O,-2);
freq_neighbours = O;

freq_neighbours(1:5,1:10)

% 2) create neighbourhood matrix for times (includes a given
% time point and the two immediately preceding and following time points)
O = ones(ntimes);
O = O - triu(O,2) - tril(O,-2);
time_neighbours = O;

% Store both matrices together in a cell array
cfg.neighbours = {freq_neighbours, time_neighbours};

% this might take a while...
[perf, result] = mv_classify(cfg, freq.trial, clabel);

% perf is a 2-D [frequencies x times] matrix of classification results
% figure
% imagesc(T,F,perf)
% colorbar
% xlabel('Time'), ylabel('Frequency')
% title('AUC at each time-frequency point including neighbours')

mv_plot_result(result, freq.time, freq.freq)

%% Classification for each time point [using channels x frequencies as features]
cfg = [];
cfg.classifier      = 'lda';
cfg.metric          = 'auc';
cfg.dimension_names = {'samples','channels','frequencies', 'time points'};

cfg.sample_dimension  = 1;
% Here, we designate both channels and frequencies as features. Since
% cfg.flatten = 1 per default, both dimension will be flattened into a
% single feature vector of length channels x frequencies
cfg.feature_dimension  = [2, 3];

[perf, result] = mv_classify(cfg, freq.trial, clabel);

% figure
% plot(T, perf)
% xlabel('Time')

mv_plot_result(result, freq.time)

%% Time-frequency classification with time generalization
% Starting from the previous example, we perform time x time classification
% (time generalization). All the channels/frequencies will be used as features, so
% the feature vector has length channels x frequencies. 
% Samples are in dimension 1 and features in
% dimension 2 and 3. The last dimension will be automatically devised as search
% dimension. However, this time we also need to indicate that we want to 
% the 4th dimension for generalization (time x time).

cfg = [];
cfg.classifier      = 'lda';
cfg.metric          = 'auc';
cfg.dimension_names = {'samples','channels','frequencies', 'time points'};

cfg.sample_dimension  = 1;
cfg.feature_dimension  = [2, 3];
cfg.generalization_dimension = 4;

cfg.repeat = 2;

[perf, result] = mv_classify(cfg, freq.trial, clabel);

% figure
% imagesc(T, F, perf)
% mv_plot_2D(perf, 'x', T, 'y', T)
% title('Time generalization using channels-x-frequencies as features')

mv_plot_result(result, freq.time, freq.time)

%% Time-frequency classification with frequency generalization
% same as previous example, but we swap frequencies and times: time points
% will serve as features, whereas we will train and test on different
% frequencies. The result is a frequency x frequency plot.

cfg = [];
cfg.classifier      = 'lda';
cfg.metric          = 'auc';
cfg.dimension_names = {'samples','channels','frequencies', 'time points'};

cfg.sample_dimension  = 1;
% Relative to the previous example, we need to swap elements in the feature 
% dimension and generalization dimension.
cfg.feature_dimension  = [2, 4];   
cfg.generalization_dimension = 3;

[perf, result] = mv_classify(cfg, freq.trial, clabel);

% figure
% mv_plot_2D(perf, 'x', F, 'y', F)
% xlabel('Test frequency [Hz]'), ylabel('Train frequency [Hz]')
% title('Frequency generalization using channels-x-times as features')

mv_plot_result(result, freq.freq, freq.freq)
