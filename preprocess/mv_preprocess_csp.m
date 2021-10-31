function [pparam, X, clabel] = mv_preprocess_csp(pparam, X, clabel)
% Calculates Common Spatial Patterns (CSP) by projecting the data onto
% n leading (largest eigenvalues) and n trailing (smallest eigenvalues) 
% eigenvectors for the generalized eigenvalue problem 
%
% lambda = (w' * C1 * w) / (w' * C2 * w)
%
% where C1 and C2 are the covariance matrices for classes 1 and 2  and w 
% is the desired spatial filter that maximizes the quotient. CSP is a
% supervised method and hence should be used within a cross-validation loop.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_csp(pparam, X, clabel)
%
%Parameters:
% X              - [... x ... x ...] data matrix. Data needs to have at
%                  least three dimensions: samples, features, and a target 
%                  dimension (e.g. time) along which the variance is
%                  calculated. If the data has more dimensions, CSP
%                  analysis will be done for each element along these extra
%                  dimensions. It is assumed that the samples dimension is
%                  the first dimension.
% clabel         - [samples x 1] vector of class labels. Works only for two
%                  classes.
%
% pparam         - [struct] with preprocessing parameters
% .n                - total number of leading and trailing components to keep
%                    (default 3). The total number of features then is 2*n
%                    (n components for class 1, another n for class 2).
% .target_dimension - dimension along which the covariance matrix is
%                     calculated (eg time or sample dimension) (default 3)
% .feature_dimension - which dimension codes the features (eg the channels)
%                     (default 2)
% .lambda            - adds a bit of identity matrix to the covariance for
%                      numerical stability: C = C + lambda * I (default
%                      10^-10)
% .calculate_variance - if 1, the variance is calculated after the CSP
%                      projection. Variance then serves as the feature for
%                      the classifier. Note that this removes the target
%                      dimension and replaces the features by variance
%                      features. Example: Assume your data is [100 samples
%                      x 50 features x 200 time points] and time points
%                      serve as target dimension. The number of CSP
%                      components is 6 (n=3). Then the output is [100
%                      samples x 6 CSP components]. Alternatively, if
%                      calculate_variance=0, the target dimension is
%                      preserved and the result is [100 samples x 6 CSP 
%                      components x 200 time points]. Note that variance is
%                      an estimate of bandpower if the signal is bandpass
%                      filtered. (default 0)
% .calculate_log     - log-transform the variance. Only applies when
%                      calculate_variance = 1 (default 1)
% .calculate_spatial_pattern - if 1 calculate the spatial pattern for each source
%                      (useful for visualization)

%
% Note: features x features covariance matrices are calculated across the
% target dimension. A covariance matrix is calculated for every element of
% any other dimension (if any) and an average covariance matrix is
% calculated. For instance, if the data is [trials x channels x times] (and
% target_dimension=3, feature_dimension=2) then single-trial covariance
% matrices are calculated across time and then averaged across trials, for 
% each class separately.
%
% Nested preprocessing: Eigenvectors of the covariance matrix are
% calculated on the train data. Both train and test data are projected onto
% the components.
%
% Reference:
% ﻿Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K. R. (2008). 
% Optimizing spatial filters for robust EEG single-trial analysis.
% IEEE Signal Processing Magazine, 25(1), 41–56. 
% https://doi.org/10.1109/MSP.2008.4408441

f = pparam.feature_dimension;
t = pparam.target_dimension;
nd = ndims(X);

if pparam.is_train_set
    % Number of samples per class
    class_ix = arrayfun(@(c) find(clabel == c), 1:2, 'UniformOutput', false);

    % all other dimensions serve as 'search' dimensions: covariance is
    % accumulated over these dimensions
    search_dim = setdiff(1:nd, [t, f]);
    
    % Create all combinations of elements in the search dimensions for looping
    if isempty(search_dim)
        % no search dimensions, we just perform cross-validation once
        dim_loop = {':'};
    else
        sz_search = size(X);
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
    
    % Split into loop for class 1 and class 2
    dim_loop1 = dim_loop(:, ismember([dim_loop{1,:}], class_ix{1}));
    dim_loop2 = dim_loop(:, ismember([dim_loop{1,:}], class_ix{2}));
    clear dim_loop

    % Start with an empty covariance matrix
    C1 = zeros(size(X, f));
    C2 = zeros(size(X, f));
    
    % if the feature dimension comes before the target dimension, we have
    % to flip the matrix for the covariance calculation. Each single trial
    % covariane matrix is normalized by its trace which increases
    % robustness (by reducing the risk of artifactual high-variance trials
    % dominating the result).
    if f < t
        for ix = dim_loop1
            tmp = cov(squeeze(X(ix{:},:))');
            C1 = C1 + tmp/trace(tmp);  
        end
        for ix = dim_loop2
            tmp = cov(squeeze(X(ix{:},:))');
            C2 = C2 + tmp/trace(tmp);  
        end
    else
        for ix = dim_loop1
            tmp = cov(squeeze(X(ix{:},:)));
            C1 = C1 + tmp/trace(tmp);
        end
        for ix = dim_loop2
            tmp = cov(squeeze(X(ix{:},:)));
            C2 = C2 + tmp/trace(tmp);
        end
    end
    
    C1 = C1/numel(class_ix{1});
    C2 = C2/numel(class_ix{2});

    % Regularize
    if pparam.lambda > 0
        C1 = C1 + pparam.lambda * eye(size(C1));
        C2 = C2 + pparam.lambda * eye(size(C2));
    end
    
    % Generalized eigenvalue decomposition gives us the filters
    [W, D] = eig(C1, C2, 'vector');
    [D, so] = sort(D,'descend');
    W = W(:, so); 

    % Spatial patterns
    if pparam.calculate_spatial_pattern
        P = inv(W)';
        P = P(:, [1:pparam.n, end-pparam.n+1:end]);
        pparam.spatial_pattern = P;
    end

    % Select n leading and trailing components
    pparam.W = W(:, [1:pparam.n, end-pparam.n+1:end]);
    pparam.eigenvalue = D([1:pparam.n, end-pparam.n+1:end]);
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
sz(end) = pparam.n*2;
X = reshape(X, sz);

% permute feature dimension back to its original position
X = permute(X, pos);

% Calculate variance and log
if pparam.calculate_variance
    X = var(X, [], t);
    if pparam.calculate_log
        X = log(X + 10^-10); % a small constant to avoid log(0)
    end
end
