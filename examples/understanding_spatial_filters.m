% UNDERSTANDING SPATIAL FILTERS
%
% This tutorial builds on the understanding_preprocessing tutorial so make
% sure you have finished that one first. 
%
% The purpose of this tutorial is to gain an understanding of spatial
% filters and how they can be used as a preprocessing tool for
% classification. Spatial filters project the data (typically EEG/MEG) into 
% a linear subspace that is optimized for a particular quantity. For 
% instance, PCA identifies a subspace that maximizes the variance of the 
% projected signals.
%
% All linear methods (ICA, beamformers and MNE, linear classifiers etc) can
% be considered as spatial filters, but in this tutorial we focus on
% spatial filters that amplify oscillatory signals of interest.
%
% An important distinction is that between spatial *filters* and spatial
% *patterns*, the former act as an inverse model that performs the
% projection onto the linear subspace, whereas the latter acts like a forward
% model that is useful for interpretation. For a detailed discussion of
% this, refer to the following paper:
%
% ﻿Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J. D., 
% Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight 
% vectors of linear models in multivariate neuroimaging. NeuroImage, 
% 87, 96–110. https://doi.org/10.1016/j.neuroimage.2013.10.067
%
% Contents:
% (1) Create simulated oscillatory EEG data
% (2) Spatio-spectral decomposition (SSD) with simulated data
% (3) Spatio-spectral decomposition (SSD) with real data
% (4) Create simulated oscillatory EEG data with 2 classes
% (5) Common Spatial Patterns (CSP) with simulated data
% (6) CSP for cross-validated classification using simulated data
% (7) CSP for cross-validated classification using real data
% (8) Using SSD+CSP for cross-validated classification
% (9) Multivariate noise normalization (MMN)
%
% Note: If you are new to working with MVPA-Light, make sure that you
% complete the introductory tutorials first:
% - getting_started_with_classification
% - getting_started_with_regression
% They are found in the same folder as this tutorial.
%

close all
clear

%% (1) Create simulated oscillatory EEG data
% We create simulated EEG data consisting of narrowband oscillations (which
% will serve as our signal) and 1/f noise broadband data (which will serve
% as our background noise).
% We use the function simulate_oscillatory_data to this end. The
% simulation provides us with a ground truth and in the next sections we
% can check the methods agains this ground truth.

% Let us generate data consisting of 100 samples (trials), 30 EEG channels,
% 512 samples per trial, and a sampling frequency of 256 Hz. 
fs = 256;
n_sample = 100;
n_channel = 30;

cfg = [];
cfg.n_sample = n_sample;
cfg.n_channel = n_channel;
cfg.n_time_point = 512;
cfg.fs = fs;

% We want to simulate 4 narrowband oscillations: a frontal theta source 
% (4-8 Hz) centered on Fz , an occipital alpha source (8-12 Hz) centered 
% on Oz, and two beta sources (14-24 Hz) centered on C3 and C4 (motor
% cortex). The number of narrowband sources is given by cfg.n_narrow, and
% their frequencies are specified by cfg.freq.
cfg.n_narrow = 4;
cfg.freq = [4 8; 
            8 12; 
            14 24;
            14 24];
% cfg.amplitude controls the SNR (where amplitude=3 means the signal has
% three times the amplitude of each broadband noise source).
cfg.amplitude = [3 3 3 3];
% simulate_oscillatory_data creates each source as a univariate signal and
% then uses a 30x1 weight vector to project the data into EEG space. If
% unspecified, the weights are randomly generated. However, in our case we
% want the weights to make sense neurophysiologically, so we define them by
% hand. To this end, we need a channel montage which we get from the
% example EEG data.
[~, ~, chans] = load_example_data('epoched1');
% We specify the weights by setting the weight for the central electrode 
% for each source to 1, setting neighbouring electrodes to 0.2, and all
% other weights to 0.
theta1 = zeros(30,1);
alpha1 = zeros(30,1);
beta1 = zeros(30,1);
beta2 = zeros(30,1);
% Let's get channel indices for Fz, Cz, C3, and C4
Fz_ix = find(ismember(chans.label,'Fz'));
Oz_ix = find(ismember(chans.label,'Oz'));
C3_ix = ismember(chans.label,'C3');
C4_ix = ismember(chans.label,'C4');
% now set the weights
theta1(Fz_ix) = 1;
theta1(ismember(chans.label,{'AF3', 'AF4' 'F3' 'F4' 'FC1' 'FC2'})) = 0.2;
alpha1(Oz_ix) = 1;
alpha1(ismember(chans.label,{'O1' 'O2'})) = 0.2;
beta1(C3_ix) = 1;
beta1(ismember(chans.label,{'FC5' 'FC1' 'CP5' 'CP1'})) = 0.2;
beta2(C4_ix) = 1;
beta2(ismember(chans.label,{'FC4' 'FC6' 'CP6' 'CP2'})) = 0.2;

% Let's plot the 4 sets of weights we have just created as topographies
cfg_plot = [];
cfg_plot.outline = chans.outline;
cfg_plot.title = {'Frontal theta' 'Occipital alpha' 'Left motor beta' 'Right motor beta'};
% cfg_plot.label = chans.label; % uncomment to see channel labels
figure
mv_plot_topography(cfg_plot, [theta1 alpha1 beta1 beta2], chans.pos);
colormap jet

% Let's store these weights in our cfg struct
cfg.narrow_weight = [theta1 alpha1 beta1 beta2];
% As background noise, we simulate 30 1/f sources. This is the default, but
% we set the parameter here for clarity
cfg.n_broad = 30;

% Now that we have all the options set, let's create the data
X = simulate_oscillatory_data(cfg);

% Let's plot some of the raw data in trial 1. Over Fz we should see an
% theta source, over Oz an alpha source. We also plot T7 which should
% contain no oscillation but only the background noise.
figure
subplot(1,3,1)
plot(squeeze(X(1,Fz_ix,:)))
title('Channel Fz [theta]')

subplot(1,3,2)
plot(squeeze(X(1,Oz_ix,:)))
title('Channel Oz [alpha]')

subplot(1,3,3)
plot(squeeze(X(1,ismember(chans.label,'T7'),:)))
title('Channel T7 [no oscillations]')

%%%%%% EXERCISE 1 %%%%%%
% To be able to better spot the simulated oscillations, increase the
% signal-to-noise ratio by setting amplitude to 10 for all sources and then
% plot the first trial again.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (2) Spatio-spectral decomposition (SSD) with simulated data
% Spatio-spectral decomposition (SSD) finds linear projections of the data 
% that identify narrowband oscillations. To this end, it tries to maximize
% power in a target signal frequency band, while simultaneously suppressing
% activity in flanking frequency bands. For details, refer to 
% mv_preprocess_ssd and the following paper:
%
% ﻿Nikulin, V. V., Nolte, G., & Curio, G. (2011). A novel method for reliable 
% and fast extraction of neuronal EEG/MEG oscillations on the basis of 
% spatio-spectral decomposition. NeuroImage, 55(4), 1528–1535. 
% https://doi.org/10.1016/j.neuroimage.2011.01.057
%
% We now perform SSD on the data created in the previos section. Since it's
% simulated data, we now the ground truth (the signal patterns and
% frequencies), and we can compare the result of SSD to it. SSD is an
% unsupervised method that seeks to extract narrowband sources from the
% mixture of sources in X. 

% Let's get the default preprocessing parameters for SSD
pparam = mv_get_preprocess_param('ssd');

% The features (channels) are in dimension 2, and the target dimension
% across which the covariance is calculated is dimension 3 (time). This is
% the default value as well, but we set it here explicitly for clarity.
pparam.feature_dimension = 2;
pparam.target_dimension = 3;

% SSD requires two versions of the data: a 'signal' version that has been
% bandpass filtered in the target frequency band, and a 'noise' version
% that has been filtered in the flanking frequency bands. For now we are
% interested in the alpha band, so 8-12 Hz will serve as our signal band,
% and 5-8 Hz and 12-15 Hz will serve as the flanking frequency bands.

X_signal = zeros(size(X));
for n = 1:n_sample
    X_tmp = bandpass(squeeze(X(n,:,:))', [8, 12], fs);
    X_signal(n,:,:) = X_tmp';
end

% For the flanking frequencies, we produce left and right flankers
% separately and then just simply add them up
X_noise = zeros(size(X));
for n = 1:n_sample
    X_left = bandpass(squeeze(X(n,:,:))', [5, 8], fs);
    X_right = bandpass(squeeze(X(n,:,:))', [12, 15], fs);
    X_noise(n,:,:) = (X_left + X_right)';
end

% Let's assign the data to the signal and noise component in pparam and run
% SSD (we use the term *source* and *component* interchangeably here). 
% Since SSD is used outside of a training loop, we need to set
% pparam.signal_train and pparam.noise_train by hand:
pparam.signal_train = X_signal;
pparam.noise_train = X_noise;
% Also we choose to calculate 10 components. Note that SSD always return the requested
% number irrespective of how many 'real' components there are in the data.
pparam.n = 10;
% Also we want to calculate the spatial pattern
pparam.calculate_spatial_pattern = 1;

% Let's run SSD. We get an updated pparam struct and the data projected
% onto the SSD components.
[pparam, X_ssd] = mv_preprocess_ssd(pparam, X);

% Looking at the size of the output we see that we now have 10 features -
% the data projected onto each of the SSD components
size(X_ssd)

% There should be one dominant alpha source (since we only simulated
% one). In a real-world application, we do not know the true number of
% source. In this case, we can plot the eigenvalues which represent a
% measure of the signal-to-noise ratio of the component. The higher it is,
% the more we can assume that there is an oscillation.
plot(pparam.eigenvalue, 'ro-')
xlabel('Component index')
ylabel('Eigenalue')
% We can see that the first component dominates the eigenvalue spectrum,
% in line with the ground truth (there is only one real component).

% Let us plot the spatial filter and the spatial patterns for the first
% component
cfg_plot = [];
cfg_plot.outline = chans.outline;
cfg_plot.title = {'Spatial filter' 'Spatial pattern'};
figure
mv_plot_topography(cfg_plot, [pparam.W(:,1) pparam.spatial_pattern(:,1)], chans.pos);
colormap jet

% The spatial pattern looks very much like our occipital alpha component
% from the previous section. Nice, we correctly recovered the source!

%%%%%% EXERCISE 2 %%%%%%
% Repeat the analysis, this time try to extra the beta sources. 
% What do you expect the eigenvalue spectrum to look like?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (3) Spatio-spectral decomposition (SSD) with real data
% We will now apply SSD to real data. Here, the ground truth is not known,
% but we can investigate whether the sources make sense
% neurophysiologically.

% Load example dataset 1
[dat, ~, chans] = load_example_data('epoched1');
X = dat.trial;
fs = 1/(dat.time(2) - dat.time(1));

% Let us target the alpha band as before.
X_signal = zeros(size(X));
for n = 1:size(X,1)
    X_tmp = bandpass(squeeze(X(n,:,:))', [8, 12], fs);
    X_signal(n,:,:) = X_tmp';
end

X_noise = zeros(size(X));
for n = 1:size(X,1)
    X_left = bandpass(squeeze(X(n,:,:))', [5, 8], fs);
    X_right = bandpass(squeeze(X(n,:,:))', [12, 15], fs);
    X_noise(n,:,:) = (X_left + X_right)';
end

pparam = mv_get_preprocess_param('ssd');
pparam.signal_train                 = X_signal;
pparam.noise_train                  = X_noise;
pparam.n                            = 30;
pparam.calculate_spatial_pattern    = 1;
[pparam, ~] = mv_preprocess_ssd(pparam, X);

% Eigenvalue plot gives us an idea on the number of components. Actually
% the real spectrum is much more nuanced than for the simulated data. We
% choose to further inspect the first 4 components.
ev = pparam.eigenvalue;
plot(ev, 'ro-')
xlabel('Component index')
ylabel('Eigenalue')

% Let us plot the spatial patterns for the first 4 components and
% let's add the eigenvalue to the title

cfg_plot = [];
cfg_plot.outline = chans.outline;
cfg_plot.nrow = 2;
cfg_plot.title = {sprintf('Pattern 1 (EV = %1.2f)', ev(1)) sprintf('Pattern 2 (EV = %1.2f)', ev(2)) sprintf('Pattern 3 (EV = %1.2f)', ev(3)) sprintf('Pattern 4 (EV = %1.2f)', ev(4))};
cfg_plot.nrow = 2; 
cfg_plot.ncol = 2;
figure
mv_plot_topography(cfg_plot, pparam.spatial_pattern(:,1:4), chans.pos);
colormap jet
% We see that the four sources with the largest signal-to-noise ratio are
% centered on parietal, occipital, and central electrode sites. This makes
% a lot of sense!

%%%%%% EXERCISE 3 %%%%%%
% Now investigate theta sources using the same approach. What kind of 
% spatial patterns do you expect to find?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (4) Create simulated oscillatory EEG data with 2 classes
% Our simulated oscillatory data so far contains all the oscillations in
% every trials. For CSP, we assume that there is two different classes and
% that different oscillations are prominent in different classes.
% To this end, let us perform another simulation with the same
% four sources (theta, alpha, two beta sources), but this time the first
% beta source in only active in class 1 trials and the second beta
% source is only active in class 2 trials.

% Let's start by copying over the cfg struct from example 1
fs = 256;

cfg = [];
cfg.n_sample = 100;
cfg.n_channel = 30;
cfg.n_time_point = 512;
cfg.fs = fs;
cfg.n_narrow = 4;
cfg.freq = [4 8; 
            8 12; 
            14 24;
            14 24];
cfg.amplitude = [3 3 3 3];
theta1 = zeros(30,1);
alpha1 = zeros(30,1);
beta1 = zeros(30,1);
beta2 = zeros(30,1);
Fz_ix = find(ismember(chans.label,'Fz'));
Oz_ix = find(ismember(chans.label,'Oz'));
C3_ix = find(ismember(chans.label,'C3'));
C4_ix = find(ismember(chans.label,'C4'));
theta1(Fz_ix) = 1;
theta1(ismember(chans.label,{'AF3', 'AF4' 'F3' 'F4' 'FC1' 'FC2'})) = 0.2;
alpha1(Oz_ix) = 1;
alpha1(ismember(chans.label,{'O1' 'O2'})) = 0.2;
beta1(C3_ix) = 1;
beta1(ismember(chans.label,{'FC5' 'FC1' 'CP5' 'CP1'})) = 0.2;
beta2(C4_ix) = 1;
beta2(ismember(chans.label,{'FC4' 'FC6' 'CP6' 'CP2'})) = 0.2;
cfg.narrow_weight = [theta1 alpha1 beta1 beta2];
cfg.n_broad = 30;

% To create multiple classes, we can set the parameter narrow_class. It is
% a [n_narrow, n_classes] binary matrix. We set it up as follows: the
% first two sources are active in both classes; the third source (row 3) is active
% only in class 1 trials; the fourth source (row 4) is active only in class 2
% trials. Note that the 1/f noise sources are active in both classes.
cfg.narrow_class = [1 1; 
                    1 1; 
                    1 0;
                    0 1];

% Let's create the data. This time, we also want the class labels as an
% output. The first half of the trials is class 1, the second half class 2
[X_sim_csp, clabel_sim_csp] = simulate_oscillatory_data(cfg);
clabel_sim_csp'

% Plotting channel C3, we can see that indeed there is a beta oscillation in
% class 1 trials but not in class 2 trials
figure
subplot(1,2,1)
plot(squeeze(X_sim_csp(1,C3_ix,:)))
title('C3 in trial 1 [class 1]')
ylim([-3,3])

subplot(1,2,2)
plot(squeeze(X_sim_csp(51,C3_ix,:)))
title('C3 in trial 51 [class 2]')
ylim([-3,3])

%% (5) Common Spatial Patterns (CSP) with simulated data
% Common Spatial Patterns (CSP) finds linear projections of the data that 
% maximize power in one class vs the other class. For details, refer to 
% mv_preprocess_csp and the following paper:
%
% ﻿Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K. R. (2008). 
% Optimizing spatial filters for robust EEG single-trial analysis.
% IEEE Signal Processing Magazine, 25(1), 41–56. 
% https://doi.org/10.1109/MSP.2008.4408441

% Let us now use the data from example 4 to calculate CSP. In this example,
% we will calculate the patterns on the full dataset and then visualize the
% spatial filters and patterns. Let's start by getting the default
% preprocessing parameters

pparam = mv_get_preprocess_param('csp');
pparam.n = 2;

% We want to visualize not only the filters but also the corresponding patterns
pparam.calculate_spatial_pattern = true;

% Let's call the function now to calculate the weights and spatial patterns
pparam = mv_preprocess_csp(pparam, X_sim_csp, clabel_sim_csp);

% Now let us the CSP components and add their corresponding eigenvalues to
% the title. One thing we notice is sine we set pparam.n we always get four
% components, 2 components with the largest eigenvalues, and 2 with the
% smallest eigenvalues. In the simulated data, there is really only one
% discriminative component for class 1 and one for class 2. Nevertheless
% the algorithm can give us any number (up to the number of channels) of
% components, but there is no guarantee that they make physiological sense.
% However, we can see that the eigenvalue for CSP 1 is much larger than for
% CSP 2, and likewise the eigenvalue for CSP 4 is much smaller than for CSP
% 3. In particular, the eigenvalues for CSP 2 and 3 are close to 1 which
% suiggests us that there is not much discriminative information in these
% components.
cfg_plot = [];
cfg_plot.outline = chans.outline;
cfg_plot.title = strcat('CSP weights #', {'1' '2' '3' '4'}', arrayfun(@(e) sprintf(' (EV = %2.2f)', e), pparam.eigenvalue, 'UniformOutput', false));
figure
mv_plot_topography(cfg_plot, pparam.W, chans.pos);
colormap jet

% The CSP weights are not really interpretable since they are optimized
% 'inverse models' that optimally suppress the noise and return the time
% series of the desired components. Spatial patterns are more interpretable
% so let's plot them. 
cfg_plot.title = strcat('Spatial pattern #', {'1' '2' '3' '4'});
figure
mv_plot_topography(cfg_plot, pparam.spatial_pattern, chans.pos);
colormap jet
% The scaling of the patterns is not very informative, but looking at the
% relative weights we can appreciate that CSP components 1 and 4 represent 
% the left and right motor cortex pattern we simulated. CSP recovered the
% motor components just as we hoped!
% Spatial patterns 2 and 3 look less physiologically plausible. In real
% data (when the true number of components is not known) we could 
% use a combination of visually inspecting the spatial patterns and
% checking the eigenvalues to determine how many meaningful components
% there are in the data.

% Note: In typical CSP applications such as motor imagery classification
% the data would be bandpass filtered in a target band (e.g. mu or beta
% band) prior to CSP. This was not necessary here since we simulated data 
% at a high SNR.

%% (6) CSP for cross-validated classification using simulated data
% So far we have used spatial filters to investigate the data and plot
% spatial patterns. However, another important use case for CSP is including it 
% in a classification pipeline. The typical use of CSP in brain-computer
% interfaces is as follows: Filter the single-trial data in the target
% frequency band, project the data into the CSP subspace, calculate
% variance (an estimate of power) and then use these power values as
% features for classification. We will start working with the simulated
% data created in section 4. We restrict it to the beta band because that's
% where we created to different sources for the two classes.

X_signal = zeros(size(X_sim_csp));
for n = 1:size(X_sim_csp,1)
    X_tmp = bandpass(squeeze(X_sim_csp(n,:,:))', [14, 24], fs);
    X_signal(n,:,:) = X_tmp';
end

% To use the CSP filtered time series as features for classification, we
% calculate the variance along the time dimension, for each CSP component
% separately. Note that variance is an estimate of signal power. The
% hypothesis is thus that one (or more) alpha sources are more active in
% one class than in the other class.
% To this end, we set calculate_variance = 1. Additionally, we log
% transform the power to get a measure akin to decibel, by using setting 
% calculate_log = 1.
pparam = mv_get_preprocess_param('csp');
pparam.calculate_variance = 1;
pparam.calculate_log = 1;

% Classification wll be performed with mv_classify. Let us use 5-fold
% cross-validation with 5 repetitions and an LDA classifier.
cfg = [];
cfg.classifier          = 'lda';
cfg.k                   = 5;
cfg.repeat              = 5;
cfg.preprocess          = 'csp';
cfg.preprocess_param    = pparam;

% We need to be careful with setting the dimensions. Normally we would set
% dimension 2 (channels) as the sole feature dimension, using dimension 3 
% (time) as the dimension to loop over. However, we are now removing 
% dimension 3 because we calculate variance across that dimension. So 
% effectively dimension 3 also serves as a feature dimension. This dimension 
% corresponds to the target_dimension in the preprocessing param. So we can
% set cfg.feature_dimension using the dimensions specified in pparam.
cfg.feature_dimension = [pparam.feature_dimension, pparam.target_dimension];
cfg.flatten_features  = 0;  % make sure the feature dimensions do not get flattened

[perf, result] = mv_classify(cfg, X_signal, clabel_sim_csp);

mv_plot_result(result)
perf

% The classification accuracy can vary due to the randomness in the data
% simulation and randomness in the cross-validation, but it should be near
% 100%. This is because we simulated the two beta sources with very high
% signal-to-noise ratio.

%%%%%% EXERCISE 4 %%%%%%
% Repeat the analysis using CSP in the theta band (4-8 Hz). 
% What performance do you epect to find?
%%%%%%%%%%%%%%%%%%%%%%%%


%% (7) CSP for cross-validated classification using real data
% We repeat the same analysis using real data this time. 
% Let us load example dataset 1 and filter it in the alpha range. We will
% then train a classifier on the alpha power values of the trials.
[dat, clabel] = load_example_data('epoched1');
X = dat.trial;
fs = 1/(dat.time(2) - dat.time(1));

% Let us target the alpha band.
X_signal = zeros(size(X));
for n = 1:size(X,1)
    X_tmp = bandpass(squeeze(X(n,:,:))', [8, 12], fs);
    X_signal(n,:,:) = X_tmp';
end

% Setup up pparam and run cross-validation as in the previous section.
pparam = mv_get_preprocess_param('csp');
pparam.calculate_variance = 1;
pparam.calculate_log = 1;
pparam.n = 10;

cfg = [];
cfg.classifier          = 'lda';
cfg.k                   = 10;
cfg.repeat              = 2;
cfg.preprocess          = 'csp';
cfg.preprocess_param    = pparam;
cfg.feature_dimension   = [pparam.feature_dimension, pparam.target_dimension];
cfg.flatten_features    = 0; 


[perf, result] = mv_classify(cfg, X_signal, clabel);

perf
% Performance is above chance albeit not great

%% (8) Using SSD+CSP for cross-validated classification
% So far we considered SSD and CSP separately, but both methods can be
% combined in a single preprocessing pipeline. There are several examples
% of this approach in the literature e.g. 
% https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5161474/
[dat, clabel] = load_example_data('epoched1');
X = dat.trial;
fs = 1/(dat.time(2) - dat.time(1));

% Remember that for SSD we need to specify both signal and noise bands. For
% CSP only the signal band will be required.
% Let's focus on alpha again as before.
X_signal = zeros(size(X));
for n = 1:size(X,1)
    X_tmp = bandpass(squeeze(X(n,:,:))', [8, 12], fs);
    X_signal(n,:,:) = X_tmp';
end

X_noise = zeros(size(X));
for n = 1:size(X,1)
    X_left = bandpass(squeeze(X(n,:,:))', [5, 8], fs);
    X_right = bandpass(squeeze(X(n,:,:))', [12, 15], fs);
    X_noise(n,:,:) = (X_left + X_right)';
end

% Let's specify the SSD parameters first. We need to set signal and noise.
% Note that unlike above we do not set signal_train and noise_train
% manually since is being done automatically during cross-validation.
ssd_param = mv_get_preprocess_param('ssd');
ssd_param.signal                 = X_signal;
ssd_param.noise                  = X_noise;
ssd_param.n                      = 10;

% The output of the SSD will still be time series. We will use these as
% inputs to CSP and then again extract the variance features for
% classification.
csp_param = mv_get_preprocess_param('csp');
csp_param.calculate_variance = 1;
csp_param.calculate_log = 1;
csp_param.n = 5;

% Let's set up the cfg struct with mostly default parameters, we only
% change the preprocessing fields. 
cfg = [];
cfg.preprocess          = {'ssd' 'csp'};
cfg.preprocess_param    = {ssd_param, csp_param};
cfg.feature_dimension   = [2 3];
cfg.flatten_features    = 0; 

% Let's run it and hope for the best.
[perf, result] = mv_classify(cfg, X_signal, clabel);

% Use SSD in conjunction with CSP does not improve performance
% significantly. The poor overall classification results could be due to
% the experimental paradigm being mostly suited for ERP analysis. Other
% paradigms involving clear oscillatory activity (such as motor imagery)
% might be better suited for this type of analysis.

% Congrats, you now mastered using SSD and CSP with MVPA-Light!

%% (9) Multivariate noise normalization (MMN)
% MMN is a pre-processing technique suggested in:
% Guggenmos, M., Sterzer, P. & Cichy, R. M. Multivariate pattern analysis 
% for MEG: A comparison of dissimilarity measures. Neuroimage 173, 434–447 (2018).
% https://www.sciencedirect.com/science/article/pii/S1053811918301411
% Supplemental material: https://ars.els-cdn.com/content/image/1-s2.0-S1053811918301411-mmc1.docx

% We will use the real EEG data for this example with a SVM classifier with
% RBF kernel. We will compare performance with and without preprocessing
% with MMN

[dat, clabel] = load_example_data('epoched3');
X = dat.trial;

cfg = [];
cfg.classifier          = 'svm';
cfg.hyperparameter      = [];
cfg.hyperparameter.kernel = 'rbf';
cfg.preprocess          = {'demean' 'mmn'};
[~, result_with_mmn] = mv_classify(cfg, X, clabel);

cfg.preprocess          = {'demean'};
[~, result_without_mmn] = mv_classify(cfg, X, clabel);

% combine both results into one plot
result_merge = mv_combine_results({result_with_mmn,result_without_mmn }, 'merge');
result_merge.plot{1}.legend_labels = {'SVM with MMN' 'SVM without MMN'};
mv_plot_result(result_merge)

% Congrats, we arrived at the end of this tutorial!

%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
% All parameters stay the same, we simply set cfg.amplitude = [10 10 10 10]
% and then plot again.
cfg = [];
cfg.n_sample = 100;
cfg.n_channel = 30;
cfg.n_time_point = 512;
cfg.fs = 256;
cfg.n_narrow = 4;
cfg.freq = [4 8; 
            8 12; 
            14 24;
            14 24];
cfg.amplitude = [10 10 10 10];
cfg.narrow_weight = [theta1 alpha1 beta1 beta2];
cfg.n_broad = 30;
X = simulate_oscillatory_data(cfg);

% Now the theta and alpha oscillations are more clearly visible, whereas T7
% still contains no oscillations as expected
figure
subplot(1,3,1)
plot(squeeze(X(1,Fz_ix,:)))
title('Channel Fz [theta]')

subplot(1,3,2)
plot(squeeze(X(1,Oz_ix,:)))
title('Channel Oz [alpha]')

subplot(1,3,3)
plot(squeeze(X(1,ismember(chans.label,'T7'),:)))
title('Channel T7 [no oscillations]')

%% SOLUTION TO EXERCISE 2
% We have to choose different signal frequency 14-24 Hz and as flanking
% frequencies for the noise component we can choose 10-14 and 24-30 Hz. 
% We have to bandpass filter signal and noise again.
X_signal = zeros(size(X));
for n = 1:n_sample
    X_tmp = bandpass(squeeze(X(n,:,:))', [14, 24], fs);
    X_signal(n,:,:) = X_tmp';
end

% For the flanking frequencies, we produce left and right flankers
% separately and then just simply add them up
X_noise = zeros(size(X));
for n = 1:n_sample
    X_left = bandpass(squeeze(X(n,:,:))', [10 14], fs);
    X_right = bandpass(squeeze(X(n,:,:))', [24 30], fs);
    X_noise(n,:,:) = (X_left + X_right)';
end

pparam = mv_get_preprocess_param('ssd');
pparam.signal_train = X_signal;
pparam.noise_train = X_noise;
pparam.n = 10;
pparam.calculate_spatial_pattern = true;
pparam = mv_preprocess_ssd(pparam, X);


plot(pparam.eigenvalue, 'ro-')
xlabel('Component index')
ylabel('Eigenalue')
% The spectrum clearly indicates that there are two components. This makes
% sense since we simulated two different beta sources.

% Let us plot the spatial patterns for these two components
cfg_plot = [];
cfg_plot.outline = chans.outline;
cfg_plot.title = {'Spatial pattern 1' 'Spatial pattern2'};
figure
mv_plot_topography(cfg_plot, [pparam.spatial_pattern(:,1:2)], chans.pos);
colormap jet
% We recovered the two beta sources as expected.

%% SOLUTION TO EXERCISE 3
% We can use exactly the same code as in section 3, the only thing we
% change is the intervals for the signal and noise frequencies.
[dat, ~, chans] = load_example_data('epoched1');
X = dat.trial;
fs = 1/(dat.time(2) - dat.time(1));

X_signal = zeros(size(X));
for n = 1:size(X,1)
    X_tmp = bandpass(squeeze(X(n,:,:))', [4, 8], fs);
    X_signal(n,:,:) = X_tmp';
end

X_noise = zeros(size(X));
for n = 1:size(X,1)
    X_left = bandpass(squeeze(X(n,:,:))', [2, 4], fs);
    X_right = bandpass(squeeze(X(n,:,:))', [8, 11], fs);
    X_noise(n,:,:) = (X_left + X_right)';
end

pparam = mv_get_preprocess_param('ssd');
pparam.signal_train                 = X_signal;
pparam.noise_train                  = X_noise;
pparam.n                            = 10;
pparam.calculate_spatial_pattern    = 1;
[pparam, X_ssd] = mv_preprocess_ssd(pparam, X);

% Plotting the eigenvalues, the result seems more clear cut than for the
% alpha band. There's one dominant component followed by a second one, then
% it starts leveling off. 
ev = pparam.eigenvalue;
plot(ev, 'ro-')
xlabel('Component index')
ylabel('Eigenalue')

% Let us plot 4 components again
cfg_plot = [];
cfg_plot.outline = chans.outline;
cfg_plot.nrow = 2;
cfg_plot.title = {sprintf('Pattern 1 (EV = %1.2f)', ev(1)) sprintf('Pattern 2 (EV = %1.2f)', ev(2))  sprintf('Pattern 3 (EV = %1.2f)', ev(3)) sprintf('Pattern 4 (EV = %1.2f)', ev(4))};
cfg_plot.nrow = 2; 
cfg_plot.ncol = 2;
figure
mv_plot_topography(cfg_plot, pparam.spatial_pattern(:,1:4), chans.pos);
colormap jet
% Components 1 and 2 have a very frontal loading. It is possible that they
% represent eye movement. Since eye movements have a large amplitude this
% would explain their relatively large eigenvalues. Component 3 looks like 
% frontal theta. Component 4 has a much broader distribution across the scalp.

% We can try plotting the time course for a given component for a single
% trial, but this is does not lead to a clear characterization of the
% component either
trial_nr = 5;
component = 1;
close all,plot(dat.time, squeeze(X_ssd(trial_nr, component, :)))
xlabel('Time')

%% SOLUTION TO EXERCISE 4
% We can use the same code as before, we only need to change the bandpass
% filter from 14-24 to 4-8.

X_signal = zeros(size(X_sim_csp));
for n = 1:size(X_sim_csp,1)
    X_tmp = bandpass(squeeze(X_sim_csp(n,:,:))', [4, 8], fs);
    X_signal(n,:,:) = X_tmp';
end

pparam = mv_get_preprocess_param('csp');
pparam.calculate_variance = 1;
pparam.calculate_log = 1;

cfg = [];
cfg.classifier          = 'lda';
cfg.k                   = 5;
cfg.repeat              = 5;
cfg.preprocess          = 'csp';
cfg.preprocess_param    = pparam;
cfg.feature_dimension   = [pparam.feature_dimension, pparam.target_dimension];
cfg.flatten_features    = 0;  % make sure the feature dimensions do not get flattened

[perf, result] = mv_classify(cfg, X_signal, clabel_sim_csp);

mv_plot_result(result)
perf
% Performance drops significantly compared to using the beta band.
% Theoretically, we would expect a performance near chance level, since
% there's no discriminative soures in this band. However, there is
% probably some 'spill-over' of the beta oscillations into other frequency
% bands due to their high signal-to-noise ratio.