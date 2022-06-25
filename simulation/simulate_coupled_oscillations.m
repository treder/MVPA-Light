function [X, cfg] = simulate_coupled_oscillations(cfg)
% Simulates two datasets with coupled oscillations. This is useful for
% multivariate regression scenarios such as CCA. This function builds on 
% simulate_oscillatory_data wherein only one set is simulated.
%
% Usage:  [X, cfg]  = simulate_coupled_oscillations(cfg)
%
% cfg          - struct with parameters:
% .n_sample         [int]   total number of samples (e.g. trials) (default 100)
% .n_channel        [int/vector] total number of channels. Can be a
%                           vector with two values for X and Y separately
% .n_time point     [int]   number of time points per sample
% .n_narrow         [int]   number of narrowband sources (default 3)
% .n_broad          [int]   number of broadband 1/f sources (default 30)
% .freq             [vector or matrix] frequency bands for narrow sources,
%                           either [minfreq, maxfreq] vector for all sources, 
%                           or a [n_narrow, 2] matrix with cutoff
%                           frequencies for each source separately
% .fs               [float] sampling frequency (default 256)
% .coupling        [matrix] [n_narrow, 2] binary matrix specifying whether
%                           a narrowband source appears in X only, Y only,
%                           or is coupled in X and Y. Coupling means that
%                           the sources are identical (100% correlated).
%                           For instance coupling = 
%                               [1 0; 
%                                1 1; 
%                                0 1]
%                           specifies 3 sources: the first one [1 0]
%                           appears only in X, and second one [1 1] appears
%                           in both X and Y (is coupled) and the third one
%                           [0 1] appears only in Y.
% .lag             [vector] for coupled sources, the time lag of the Y
%                           source (in samples) wrt the X source
% .amplitude        [vector/matrix] specifies the amplitude each narrowband
%                           signal is multiplied with. A [n_narrow,2]
%                           matrix can be provided instead of the vector to
%                           specify different amplitudes for X and Y
% .narrow_weight    [matrix] [n_channel, n_narrow] matrix of weights that
%                           show how a source projects into sensor space.
%                           If not provided, random weights are created and
%                           set to norm 1 each
% .broad_weight     [matrix] [n_channel, n_broad] matrix for the broadband
%                           sources.
% .sensor_noise     [float] amplitude of uncorrelated Gaussian sensor noise 
%                           that is added to each channel (default 0.01)
%
% Returns:
% X         - [n_sample x n_channel x n_time_point] array of simulated
%             EEG data with all noise and signal sources mixed together.
%             Roughly speaking, we have X = narrow + broad + sensor_noise.
% Y         - second set of data with some sources being coupled with the
%             source in Y
% clabel    - vector of class labels
% cfg       - cfg struct with all parameters and narrow and broadband sources 

mv_set_default(cfg,'n_sample',100);
mv_set_default(cfg,'n_time_point',512);
mv_set_default(cfg,'n_channel',32);
mv_set_default(cfg,'n_narrow',3);
mv_set_default(cfg,'n_broad',30);
mv_set_default(cfg,'freq',[8 12]);
mv_set_default(cfg,'fs', 256);
mv_set_default(cfg,'narrow_class', []);
mv_set_default(cfg,'amplitude', ones(cfg.n_narrow, 1));
mv_set_default(cfg,'narrow_weight',[]);
mv_set_default(cfg,'broad_weight',[]);
mv_set_default(cfg,'sensor_noise',0.01);

assert(numel(cfg.freq)==2 || all(size(cfg.freq) == [cfg.n_narrow, 2]), 'cfg.freq must either have two elements or be a [n_signal, 2] matrix' )
assert(all(cfg.freq(:)>=1) && all(cfg.freq(:)<=cfg.fs/2), sprintf('cfg.freq must be in the range [1.0, fs/2] but it is in the range [%2.1f, %2.1f]', min(cfg.freq(:)), max(cfg.freq(:))))
assert(isempty(cfg.narrow_class) || size(cfg.narrow_class,1) == cfg.n_narrow, 'narrow_class must either be empty or a [n_signal, n_class] indicator matrix for multiple classes' )
assert(any(numel(cfg.amplitude)==[1 cfg.n_narrow]), 'amplitude must be a scalar or a n_signal vector')
assert(isempty(cfg.narrow_weight) || all(size(cfg.narrow_weight) == [cfg.n_channel, cfg.n_narrow]), 'narrow_weight must be empty or a [n_channel, n_narrow] matrix')
assert(isempty(cfg.broad_weight) || all(size(cfg.broad_weight) == [cfg.n_channel, cfg.n_broad]), 'broad_weight must be empty or a [n_channel,n_broad] matrix')

if numel(cfg.freq)==2, cfg.freq = repmat(cfg.freq(:)', [cfg.n_narrow, 1]); end
if isempty(cfg.narrow_class), cfg.n_class = 1; else, cfg.n_class = size(cfg.narrow_class,2); end

assert(cfg.n_sample/cfg.n_class == round(cfg.n_sample/cfg.n_class), 'n_sample must be divisible by the number of classes' )

fprintf('Creating [%d samples x %d chans x %d time points] simulated data comprising %d signals and %d 1/f noise sources and %d classes.\n', ...
    cfg.n_sample, cfg.n_channel, cfg.n_time_point, cfg.n_narrow, cfg.n_broad, cfg.n_class)


%% Create class labels
clabel = arrayfun( @(x) ones(cfg.n_sample/cfg.n_class,1) * x, 1:cfg.n_class, 'Un', 0);
clabel = cat(1, clabel{:});

%% Create weights (= forward model from source to sensor space) if not provided
if isempty(cfg.narrow_weight)
    cfg.narrow_weight = randn(cfg.n_channel, cfg.n_narrow);
    for ix = 1:cfg.n_narrow
        cfg.narrow_weight(:, ix) = cfg.narrow_weight(:, ix) / norm(cfg.narrow_weight(:, ix));
    end
end

if isempty(cfg.broad_weight)
    cfg.broad_weight = randn(cfg.n_channel, cfg.n_broad);
    for ix = 1:cfg.n_broad
        cfg.broad_weight(:,ix) = cfg.broad_weight(:,ix) / norm(cfg.broad_weight(:,ix));
    end
end

%% Create narrowband sources
narrow = randn(cfg.n_sample * cfg.n_time_point, cfg.n_narrow);

for ix=1:cfg.n_narrow
    narrow(:, ix) = bandpass(narrow(:,ix), cfg.freq(ix,:), cfg.fs) * cfg.amplitude(ix);
end

narrow = reshape(narrow, [cfg.n_time_point, cfg.n_sample, cfg.n_narrow]);
narrow = permute(narrow, [2, 3, 1]);

if cfg.n_class > 1
    % zero out signals when they don't appear in particular class trials
    for c = 1:cfg.n_class
        for n = 1:cfg.n_narrow
            if cfg.narrow_class(n, c) == 0
                narrow( clabel==c, n, :) = 0;
            end
        end
    end
end

%% Create broadband 1/f sources
broad = randn(cfg.n_sample * cfg.n_time_point, cfg.n_broad);

% adapted from: https://uk.mathworks.com/matlabcentral/answers/344786-generation-of-1-f-noise-using-matlab#answer_270725
fv = linspace(0, 1, 20);                                % Normalised Frequencies
a = 1./(1 + fv*2);                                      % Amplitudes Of ‘1/f’
b = firls(42, fv, a);  

for ix=1:cfg.n_broad
    broad(:, ix) = filtfilt(b, 1, broad(:, ix));
end

broad = reshape(broad, [cfg.n_time_point, cfg.n_sample, cfg.n_broad]);
broad = permute(broad, [2, 3, 1]);

%% Project narrow and broad sources into sensors
X = zeros(cfg.n_sample, cfg.n_channel, cfg.n_time_point);

for ix = 1:cfg.n_sample
    X(ix,:,:) = cfg.narrow_weight * squeeze(narrow(ix, :, :)) + cfg.broad_weight * squeeze(broad(ix, :, :));
end

%% Add a bit of uncorrelated sensor space noise too
X = X + randn(size(X)) * cfg.sensor_noise;

cfg.narrow = narrow;
cfg.broad = broad;
