% Unit testing of mv_ssd_wave
clear all

% Reset random number generator
rng(2);

% Set parameters
nChan = 64;
fsample = 256;
nNoiseSources = 100;

% SNR is the multiplier for the signal amplitude
SNR = 2;

% Time axis
nTime = fsample*60*2;       % simulate 2 mins of data
time = linspace(1/fsample, nTime/fsample, nTime);

%% Signal: There's 3 signal components in the alpha range
nSig = 3;

% Forward model
forward_sig = zeros(nChan,nSig);
forward_sig(1:5,1) = 5:-1:1;   % Source 1 loads on channels 1-5
forward_sig(3:7,2) = 1:5;       % Source 2 loads on channels 2 and 3
forward_sig(10:11,3) = 5;       % Source 3 loads on channels 10 and 11

% Filter signal in the alpha range using Fieldtrip
sig_timeseries = randn(nSig, nTime);

sig_timeseries = ft_preproc_bandpassfilter(sig_timeseries,fsample,[8 12]);

% Decorrelate the time series using PCA
[~,sig_timeseries]= pca(sig_timeseries');
sig_timeseries = sig_timeseries';

sig_timeseries = zscore(sig_timeseries')';

% Scale the three sources differently so that they should come out in order
% when SSD is applied
weights = diag([3 2 1]);
 
%% Noise

% Create random forward models for the signal and noise sources
forward_noise = zscore(randn(nChan, nNoiseSources));

% Create random broadband time series
noise_timeseries = randn(nNoiseSources, nTime);

%% Project signal and noise into sensor space

% Project signals
X_sig = forward_sig * sig_timeseries;

% Project noise
X_noise = forward_noise * noise_timeseries;

% Add signal and noise together
X = SNR * X_sig + X_noise;

%% Create Fieldtrip struct for data
dat = struct();
dat.fsample     = fsample;
dat.label       = arrayfun(@(x) ['Chan' num2str(x)], 1:nChan,'un',0);
dat.dimord      = 'rpt_chan_time';
dat.trial(1,:,:)= X;       
dat.time         = time;

%% Test SSD
sel_sam = (256:nTime-256);% leave out 1s  of samples to prevent border effects

cfg = [];
cfg.nCycle      = 5;
cfg.nComp       = 5;
cfg.toi         = dat.time(sel_sam); 
cfg.foi         = 10;
[wave,W,D,A] = mv_ssd_wave(cfg, dat);

ssd_x = real(squeeze(wave.trial));  % SSD time series of components

%% Plot ERP and classification result - do they match in terms of time?
close all
nCol = 3;
nRow = 2;
n=0;

% Sensor covariance matrix
n=n+1;
subplot(nRow,nCol,n)
imagesc(cov(X'))
title('Sensor signal covariance matrix')
colorbar('Location','SouthOutside')


% SSD spatial pattern
n=n+1;
subplot(nRow,nCol,n)
imagesc(A)
title('SSD spatial patterns (in cols)')
colorbar('Location','SouthOutside')

% SSD spatial filter
n=n+1;
subplot(nRow,nCol,n)
imagesc(W(:,1:3))
title('SSD spatial filters (in cols)')
colorbar('Location','SouthOutside')

% Correlation between SSD components and true signal
n=n+1;
subplot(nRow,nCol,n)
imagesc(corr(ssd_x', sig_timeseries(:,sel_sam)'))
title('Correlation SSD <-> true signal')
colorbar('Location','SouthOutside')

% Eigenvalue
n=n+1;
subplot(nRow,nCol,n)
bar(D)
title('Eigenvalues')
