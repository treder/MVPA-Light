% Unit testing of the mv_classify_across_time function.
% Re-run as a sanity check if the code is significantly changed.
clear all

classifier = 'lda';
% classifier = 'ensemble';

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

% Signal
% We model the signal as a cosine window in the  [0.5, 1] interval and
% 0 else in class 1. In class 2, it is always zero
forward_sig = zscore(randn(nChan, 1)); 
sig_timeseries = zeros(nTime,1);
idx_sig = find(time >= 0.5  & time <= 1);
sig_timeseries(idx_sig) = cos( linspace(-pi,pi,numel(idx_sig))) + 1;

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

%% Test mv_classify_across_time

lambda = 'auto';

% Configuration for time classification with cross-validation
ccfg =  [];
ccfg.CV         = 'kfold';
ccfg.K          = 5;
ccfg.repeat     = 5;
ccfg.verbose    = 1;
ccfg.balance    = 'undersample'; % 'oversample' 'undersample'

if strcmp(classifier,'lda')
    ccfg.classifier = 'lda';
    ccfg.param      = struct('lambda',lambda);
    
elseif strcmp(classifier,'ensemble')
    ensemble_cfg= [];
    ensemble_cfg.learner        = 'lda';
    ensemble_cfg.nFeatures      = 0.3;
    ensemble_cfg.nLearners      = 100;
    ensemble_cfg.simplify       = 1;
    ensemble_cfg.learner_param  = struct('lambda',lambda);
    
    ccfg.classifier = 'ensemble';
    ccfg.param      = ensemble_cfg;
end

% Run classification
acc = mv_classify_across_time(ccfg,X,labels);

%% Plot ERP and classification result - do they match in terms of time?
figure
nCol = 4;

% Plot also the simulated signal
subplot(1,nCol,1)
plot(time,sig_timeseries)
title('True signal'),xlabel('Time'), ylabel('Classification accuracy')

% Plot the simulated noise separately for one trial
subplot(1,nCol,2)
plot(time, squeeze(X_noise(1,:,:)) )
title('Noise'),xlabel('Time'), ylabel('Classification accuracy')

% ERP
subplot(1,nCol,3)
plot(time,ERP)
title('ERPs of all channels (signal+noise)'),xlabel('Time'), ylabel('Amplitude')

% Classification result
subplot(1,nCol,4)
plot(time,acc)
title('Classification result (matches true signal?)'),xlabel('Time'), ylabel('Classification accuracy')
