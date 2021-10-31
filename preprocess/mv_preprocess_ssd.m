function [pparam, X, clabel] = mv_preprocess_ssd(pparam, X, clabel)
% Calculates Spatio-Spectral Decomposition (SSD) (Nikulin et al., 2012).
%
% lambda = (w' * Cs * w) / (w' * Cn * w)
%
% where Cs and Cn are the covariance matrices for signal and noise and w 
% is the desired spatial filter that maximizes the quotient. In contrast to
% CSP, SSD is unsupervised (it does not use the class labels).
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_ssd(pparam, X, <clabel>)
%
%Parameters:
% X              - [... x ... x ...] data array. Data needs to have at
%                  least three dimensions: samples, features, and a target 
%                  dimension (e.g. time) along which the variance is
%                  calculated. If the data has more dimensions, SSD
%                  analysis will be done for each element along these extra
%                  dimensions. It is assumed that the samples dimension is
%                  the first dimension.
% clabel         - [samples x 1] vector of class labels (not used, can be omitted)
%
% pparam         - [struct] with preprocessing parameters
% .n             - total number of components to keep (default 5). 
% .signal        - signal data represented by an array that has the
%                  same number of samples and features as X. Note: during 
%                  cross-validation the field pparam.signal_train is
%                  created from pparam.signal. It contains the training
%                  trials. If SSD is to be used directly,
%                  pparam.signal_train needs to be set by hand (e.g.
%                  pparam.signal_train = pparam.signal).
% .noise         - noise data represented by an array that has the
%                  same shape as signal.  Note: during 
%                  cross-validation the field pparam.noise_train is
%                  created from pparam.noise. It contains the training
%                  trials. If SSD is to be used directly,
%                  pparam.noise_train needs to be set by hand (e.g.
%                  pparam.noise_train = pparam.noise).
%
% Usually the signal is data bandpass filtered in a narrow band
% (e.g. 8-12 Hz), whereas the noise would be the flanking frequencies (e.g.
% 6-8 and 12-14 Hz). If signal and noise are functions, they would be the
% function performing the bandpass filtering.
%
% .target_dimension - dimension along which the covariance matrix is
%                     calculated (eg time or sample dimension) (default 3)
% .feature_dimension - which dimension codes the features (eg the channels)
%                     (default 2)
% .lambda            - adds a bit of identity matrix to the covariance for
%                      numerical stability: C = C + lambda * I (default
%                      10^-10)
% .calculate_variance - if 1, the variance is calculated after the 
%                      projection. Variance then serves as the feature for
%                      the classifier. Note that this removes the target
%                      dimension and replaces the features by variance
%                      features. Example: Assume your data is [100 samples
%                      x 50 features x 200 time points] and time points
%                      serve as target dimension. The number of 
%                      components is n=5. Then the output is [100
%                      samples x 5 SSD components]. Alternatively, if
%                      calculate_variance=0, the target dimension is
%                      preserved and the result is [100 samples x 5 SSD 
%                      components x 200 time points]. Note that variance is
%                      an estimate of bandpower if the signal is bandpass
%                      filtered. (default 0)
% .calculate_log     - if 1 and calculate_variance=1, then the variance is
%                      log-transformed (default 1).
% .calculate_spatial_pattern - if 1 calculate the spatial pattern for each source
%                      (useful for visualization)
% .data              - field containing the cell array {'signal' 'noise'}
%                      which tells MVPA-Light that the data in both fields
%                      should be subselected in cross-validation
%                      (normally this field should not be changed)
%
% Note: features x features covariance matrices are calculated across the
% target dimension. A covariance matrix is calculated for every element of
% any other dimension (if any) and an average covariance matrix is
% calculated. For instance, if the data is [trials x channels x times] (and
% target_dimension=3, feature_dimension=2) then single-trial covariance
% matrices are calculated across time and then averaged across trials.
%
% Nested preprocessing: Eigenvectors of the covariance matrix are
% calculated on the train data. Both train and test data are projected onto
% the components.
%
% Reference:
% ﻿Nikulin, V. V., Nolte, G., & Curio, G. (2011). A novel method for reliable 
% and fast extraction of neuronal EEG/MEG oscillations on the basis of 
% spatio-spectral decomposition. NeuroImage, 55(4), 1528–1535. 
% https://doi.org/10.1016/j.neuroimage.2011.01.057

f = pparam.feature_dimension;
t = pparam.target_dimension;
nd = ndims(X);

if pparam.is_train_set

    % all other dimensions serve as 'search' dimensions: covariance is
    % accumulated over these dimensions
    search_dim = setdiff(1:nd, [t, f]);
    
    % Create all combinations of elements in the search dimensions for looping
    if isempty(search_dim)
        % no search dimensions, we just perform cross-validation once
        dim_loop = {':'};
    else
        sz_search = size(pparam.signal_train);
        sz_search = sz_search(search_dim);
        len_loop = prod(sz_search);
        dim_loop = zeros(nd, len_loop);
        for rr = 1:numel(sz_search)  % row
            seq = mv_repelem(1:sz_search(rr), prod(sz_search(1:rr-1)));
            dim_loop(search_dim(rr), :) = repmat(seq, [1, len_loop/numel(seq)]);
        end
        % to use dim_loop for indexing, we need to convert it to a cell array
        dim_loop = num2cell(dim_loop);
        % we only need to replace the feature and target rows with {:}
        % operators 
        dim_loop(f,:) = {':'};
        dim_loop(t,:) = {':'};
    end
    
    % Start with an empty covariance matrix
    C_signal = zeros(size(X, f));
    C_noise = zeros(size(X, f));
    
    % if the feature dimension comes before the target dimension, we have
    % to flip the matrix for the covariance calculation. Each single trial
    % covariane matrix is normalized by its trace which increases
    % robustness (by reducing the risk of artifactual high-variance trials
    % dominating the result).
    if f < t
        for ix = dim_loop
            tmp = cov(squeeze(pparam.signal_train(ix{:},:))');
            C_signal = C_signal + tmp/trace(tmp);  
            tmp = cov(squeeze(pparam.noise_train(ix{:},:))');
            C_noise = C_noise + tmp/trace(tmp);  
        end
    else
        for ix = dim_loop
            tmp = cov(squeeze(pparam.signal_train(ix{:},:)));
            C_signal = C_signal + tmp/trace(tmp);
            tmp = cov(squeeze(pparam.noise_train(ix{:},:)));
            C_noise = C_noise + tmp/trace(tmp);
        end
    end
    
    C_signal = C_signal/size(X,1);
    C_noise = C_noise/size(X,1);

    % Regularize
    if pparam.lambda > 0
        C_signal = C_signal + pparam.lambda * eye(size(C_signal));
        C_noise = C_noise + pparam.lambda * eye(size(C_noise));
    end
    
    % Generalized eigenvalue decomposition gives us the filters
    [W, D] = eig(C_signal, C_noise, 'vector');
    [D, so] = sort(D,'descend');
    W = W(:, so); 

    % Spatial patterns
    if pparam.calculate_spatial_pattern
        P = inv(W)';
        P = P(:, 1:pparam.n);
        pparam.spatial_pattern = P;
    end

    % Select n leading and trailing components
    pparam.W = W(:, 1:pparam.n);
    pparam.eigenvalue = D(1:pparam.n);
end

% reshape data matrix such that it is [... x features] because then we can
% project it onto the components with a single matrix
% multiplication. To this end, first permute the features dimension to be
% the last dimension, then reshape
pos = 1:nd;
pos(f) = nd;
pos(nd) = f;
X = permute(X, pos);
sz = size(X); % remember size
X = reshape(X, [], size(X, nd));

% Project data into subspace
X = X * pparam.W;

% Undo reshape
sz(end) = pparam.n;
X = reshape(X, sz);

% permute feature dimension back to its original position
X = permute(X, pos);

% Calculate variance and log
if pparam.calculate_variance
    X = var(X, [], t);
    if pparam.calculate_log
        X = log(X + 10^-10); % add small constant to avoid log(0)
    end
end
