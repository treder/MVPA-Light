function X = simulate_erp_peak(n_trials, n_time_points, pos, width, amplitude, weight, shape, scale)
% Simulates ERP data.
%
% Usage:  [X,y] = simulate_erp_peak(n_samples, n_time_points, pos, width,
% amplitude, shape)
%
% Parameters:
% n_trials        [int]  total number of samples (e.g. trials)
% n_time points   [int]  length of trial in number of time points
% pos             [int or vector or matrix] center position of erp peak given as time index.  A
%                        vector can be provided to simulate positions for
%                        multiple peaks. Alternatively a [n_trials x
%                        n_peaks] matrix can be provided to simulate a
%                        different pos in every trial.
% width           [int or vector or matrix] width of peak in time points. A
%                        vector can be provided to simulate different width for
%                        different peaks. Alternatively a [n_trials x
%                        n_peaks] matrix can be provided to simulate a
%                        different width in every trial.
%                        For gaussian peaks, width controls the standard
%                        deviation, for cosine and triangle it is the
%                        distance between the maximum and 0
% amplitude       [int or vector or matrix] amplitude of peak. A
%                        vector can be provided to simulate different amplitudes for
%                        different peaks. Alternatively a [n_trials x
%                        n_peaks] matrix can be provided to simulate a
%                        different amplitude in every trial.
% weight          [vector] the 1D ERP signal can be projected to a multivariate 
%                        signal [n_trials x n_channels x n_time_points],
%                        where n_channels is the number of elements in
%                        weights. The weights specify the magnitude by
%                        which the signal is multiplied for each channel.
%                        This way a signal with a certain spatial focus
%                        (e.g over Cz) can be simulated. (default [] i.e.
%                        the signal is univariate)
% shape           [str]  shape of peak 'gaussian' 'cosine' or 'triangle'
%                        (default gaussian). 
% scale           [double] variance of random Gaussian noise added to the signal 
%                        If 0, data is perfectly defined by the functional
%                        relationship. Set to larger valurs to make the data more
%                     fuzzy (default 0)
%
% Returns:
% X         - [n_samples x n_time_points] array (if weights=[]) or a
%             multivariate [n_samples x n_channels x n_time_points] signal
%             if weight vector is provided

if nargin<5, amplitude = 1; end
if nargin<6, weight =[]; else, weight = weight(:); end
if nargin<7, shape ='gaussian'; end
if nargin<8, scale = 0; end

check_assertions(pos, 'pos');
check_assertions(width, 'width')
check_assertions(amplitude, 'amplitude')

% decide whether output is 2D or 3D
if isempty(weight)
    X = zeros(n_trials, n_time_points);
else
    n_channels = numel(weight);
    X = zeros(n_trials, n_channels, n_time_points);
end

if ismatrix(pos), n_peaks = size(pos,2);
else, n_peaks = numel(pos);
end

% repeat pos, width and amplitude into a matrix if necessary
pos = expand_to_matrix(pos);
width = expand_to_matrix(width);
amplitude = expand_to_matrix(amplitude);

for n=1:n_trials
    signal = zeros(1, n_time_points);
    
    % create peaks for current trial
    for p=1:n_peaks
        signal = signal + create_peak(signal, pos(n,p), width(n,p), amplitude(n,p));
    end
    
    % turn to multivariate
    if ~isempty(weight)
        X(n,:,:) = weight * signal;
    else
        X(n,:) = signal;
    end
end


%% Add noise
X = X + randn(size(X)) * scale;


%% helper functions

    function signal = create_peak(signal, p, w, a)
        switch(shape)
            case 'gaussian'
                signal = normpdf(1:n_time_points, p, w);
                signal = signal/max(signal) * a;
                
            case 'cosine'
                sig = a * cos(linspace(-pi/2, pi/2, 2*w+1));
                r_start = max(1, p-w); % make sure we don't shoot out beyond the boundaries
                r_end = min(n_time_points, p+w);
                signal(r_start, r_end) = signal(r_start, r_end) + sig(r_start, r_end);
                
            case 'triangle'
                slope = linspace(0, a, w+1);
                r_start = min(w, p-1); % make sure we don't shoot out beyond the boundaries
                r_end = min(w, n_time_points-p); % make sure we don't shoot out beyond the boundaries
                signal(p-r_start:p) = slope(end-r_start:end);
                signal(p:p+r_end) = slope(end:-1:end-r_end);
        end
    end

    function check_assertions(val, name)
        if numel(val)>1 && ~isvector(val)
%             if isvector(val), assert(numel(val)==n_trials, sprintf('Number of elements in ''%s'' must be equal to n_trials', name));
%             else, assert(size(val,1)==n_trials, sprintf('Number of rows in ''%s'' must be equal to n_trials', name));
%             end
            assert(size(val,1)==n_trials, sprintf('Number of rows in ''%s'' must be equal to n_trials', name));
        end
    end

    function val = expand_to_matrix(val)
        if isvector(val)
            if numel(val)>1, val = repmat(val(:)', n_trials, 1);
            else, val = val * ones(n_trials, n_peaks);
            end
        end
    end


end
