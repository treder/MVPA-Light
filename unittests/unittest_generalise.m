% Unit testing of the mv_classify_timextime function
clear all

% Reset random number generator
rng(1);

% Set parameters
nChan = 64;
nEpochsPerClass = 100;
labels = [ones(nEpochsPerClass,1); -1*ones(nEpochsPerClass,1)];
nNoiseSources = 100;

% SNR is the multiplier for the signal amplitude
SNR = 2;

% Time axis
nTime = 512;
time = linspace(-0.5, 1.5, nTime);

% Signal: There's 3 signal components, each modelled by a cosine window. 
% 1 - Peak in the interval [0.2, 0.3]
% 2 - Peak in the interval [0.4, 0.8]
% 3 - Peak in the interval [1.2, 1.5]
% Furthermore, the peaks come from 2 different sources. Peak 1 comes from
% source 2. Peaks 2 and 3 come from source 2. In the time x
% time generalisation, the classifier should hence generalise between 
% peaks 2 and 3, but not from peak 1 to the other peaks
forward_sig = zscore(randn(nChan, 2));  % we need two spatial patterns
sig_timeseries = zeros(nTime,2);

idx_peak1 = find(time >= 0.2  & time <= 0.3);
idx_peak2 = find(time >= 0.4  & time <= 0.8);
idx_peak3 = find(time >= 1.2  & time <= 1.5);

sig_timeseries(idx_peak1,1) = cos( linspace(-pi,pi,numel(idx_peak1))) + 1;
sig_timeseries(idx_peak2,2) = cos( linspace(-pi,pi,numel(idx_peak2))) + 1;
sig_timeseries(idx_peak3,2) = cos( linspace(-pi,pi,numel(idx_peak3))) + 1;

% Create forward models for the signal and noise sources
forward_noise = zscore(randn(nChan, nNoiseSources));
noise_timeseries = randn(nNoiseSources, nTime*nEpochsPerClass*2);

%% Project signal and noise into sensor space

% Project signal for each class
X_class1 = permute(repmat(forward_sig * sig_timeseries',[1 1 nEpochsPerClass ]), [3 1 2]);
X_class2 = zeros(size(X_class1));

% Project noise
X_noise = forward_noise * noise_timeseries;
X_noise = reshape(X_noise, nChan, nTime, []);
X_noise = permute(X_noise, [3 1 2]);

% Add signal and noise together
X = SNR * cat(1,X_class1, X_class2) + X_noise;

%% Calculate ERP
ERP = squeeze(mean(X,1));

%% Test mv_classify_timextime
cfg = [];
cfg.classifier      = 'lda';
cfg.param           = struct('gamma','auto');
cfg.CV              = 'kfold';
cfg.K               = 4;
cfg.repeat          = 1;
cfg.normalise       = 'zscore';
cfg.verbose         = 1;

acc = mv_classify_timextime(cfg,X,labels);


%% Plot ERP and classification result - do they match in terms of time?
close all
nCol = 4;

% Plot also the simulated signal
subplot(1,nCol,1)
plot(sig_timeseries)
% plot(time,sig_timeseries)
title('Simulated signal'),xlabel('Time'), ylabel('Classification accuracy')

% Plot the simulated noise separately for one trial
subplot(1,nCol,2)
plot(time, squeeze(X_noise(1,:,:)) )
title('Simulated noise'),xlabel('Time'), ylabel('Classification accuracy')

% ERP
subplot(1,nCol,3)
plot(time,ERP)
title('ERPs of all channels (signal+noise)'),xlabel('Time'), ylabel('Amplitude')

% Classification result
subplot(1,nCol,4)
imagesc(time,time,acc)
title('Classification result'),xlabel('Time'), ylabel('Time')
colorbar('Location','SouthOutside')
% Set colorlimit symmetric about 0.5
cl = max(abs(get(gca,'CLim') - 0.5 )); 
set(gca,'CLim', [-cl cl] + 0.5)