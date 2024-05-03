function [pparam, X, clabel] = mv_preprocess_mmn(pparam, X, clabel)
% Applies Multivariate Noise Normalization (MMN) (Guggenmos et al., 2018).
% Data is denoised with an estimate of the noise covariance:
%
%               X_denoised = C^(-1/2) * X
%
% where C is the covariance matrix and C^(-1/2) is its inverse square-root.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_mmn(pparam, X, clabel)
%
%Parameters:
% X              - [... x ... x ...] data matrix. Data needs to have at
%                  least two dimensions: samples and features. It is
%                  assumed that sample dimension is the 1st dimension.
% clabel         - [samples x 1] vector of class labels. 
%
% pparam         - [struct] with preprocessing parameters
% .target_dimension - dimension(s) along which the covariance matrix C is
%                     estimated (eg time) (default []).
%                     For instance, assume the data is [samples x features
%                     x time points]. Since feature_dimension = 2 in this case
%                     and samples form the 1st dimension, we have two
%                     choices for target_dimension:
%                     target_dimension=[]: covariance is calculated separately 
%                       for every timepoint. The resultant rank 1 covariance
%                       matrices are then averaged across samples. A total
%                       of T covariance matrices are calculated where T is
%                       the number of time points. (this is the "time point
%                       method" in Guggenmos et al)
%                     target_dimension=3: covariance is accumulated across
%                       time points and then averaged across samples. Only
%                       a single covariance matrix is calculated this way. 
%                       (this is the "epoch method" in Guggenmos et al)
% .target_indices   - specify indices based on which the covariance is
%                     calculated. This is useful if covariance is
%                     calculated over the pre-stimulus baseline. Eg, if the 
%                     first 100 time points form the baseline, setting 
%                     target_indices = 1:100 makes sure that only the
%                     baseline is used. The default {':'} is to use all
%                     indices.
% .feature_dimension - which dimension codes the features (eg the channels)
%                     (default 2).
% .sample_dimension - which dimension codes the samples (default 1).
% .lambda        - Regularization of the covariance matrix. The regularization parameter ranges 
%                  from 0 to 1 (where 0=no regularization and 1=maximum
%                  regularization). If 'auto' then the shrinkage 
%                  regularization parameter is calculated automatically 
%                  using the Ledoit-Wolf formula(function cov1para.m).
%                  Default 'auto'.
%
% Reference:
% Guggenmos, M., Sterzer, P. & Cichy, R. M. Multivariate pattern analysis 
% for MEG: A comparison of dissimilarity measures. Neuroimage 173, 434–447 (2018).
% https://www.sciencedirect.com/science/article/pii/S1053811918301411
% Supplemental material: https://ars.els-cdn.com/content/image/1-s2.0-S1053811918301411-mmc1.docx
 
s = pparam.sample_dimension;
f = pparam.feature_dimension;
t = pparam.target_dimension;
nd = ndims(X);

if ~iscell(pparam.target_indices)
    pparam.target_indices = {pparam.target_indices};
end

if nd > 2 
    % reshape X and create dim_loop

    % define non-sample/feature dimension(s) that will be used for search/looping
    search_dim = setdiff(1:ndims(X), [s, f, t]);
    sz_search = size(X);
    sz_search = sz_search(search_dim);

    % To be able to deal with arrays of any dimension and any number of
    % search dims (dimensions across which we loop) and target dims
    % (dimensions across we accumulate the covariance) we permute the array
    % as follows:
    % dim 1: samples
    % dims 2, 3, ..., m: search dims
    % dims m+1, m+2, ..., : target dims
    % last dim: features
    new_dim_order = [s, search_dim, t, f];
    X = permute(X, new_dim_order);

    % after permuting, the t's come after s + search_dim
    t = (1:numel(t)) + 1 + numel(search_dim);

    n_samples = size(X,1);
    if numel(t) > 1
        % if there is multiple target dimensions, we reshape them into one
        error('NOT IMPLEMENTED YET: if there is multiple target dimensions, we reshape them into one')
    end

    % Create all combinations of elements in the search dimensions for looping
    len_loop = prod(sz_search);
    dim_loop = zeros(numel(sz_search), len_loop);
%     dim_loop = zeros(max(1,numel(sz_search)), len_loop);
    for rr = 1:numel(sz_search)  % row
        seq = mv_repelem(1:sz_search(rr), prod(sz_search(1:rr-1)));
        dim_loop(rr, :) = repmat(seq, [1, len_loop/numel(seq)]);
    end
    % to use dim_loop for indexing, we need to convert it to a cell array
    dim_loop = num2cell(dim_loop);
    % we only need to replace the feature and target rows with {:} operators
    %         dim_loop(f,:) = {':'};  % features
    
%     if ~isempty(t)
%         dim_loop(t-1,:) = {':'};
% %         if isempty(pparam.target_indices)
% %             % note: we use t-1 because the first dimension (samples) is
% %             % looped over separately
% %             dim_loop(t-1,:) = {':'};
% %         else
% %             dim_loop(t-1,:) = {pparam.target_indices};
% %         end
%     end
end

if pparam.is_train_set
    if nd == 2
        % --- 2D case ---
        % data is just [samples x features] so we just calculate one
        % covariance matrix
        C = cov(X);

        % Regularization
        if ischar(pparam.lambda) && strcmp(pparam.lambda,'auto')
            lambda = LedoitWolfEstimate(X - mean(X), 'primal');
        end
        C = (1-lambda)* C + lambda * eye(size(C,1)) * trace(C)/size(C,1);

        % Inverse square root
        C_invsqrt = inv(sqrtm(C));
        clear C

    else % --- >2 dimensions ---
        % create empty MMN matrices
        C_invsqrt = zeros([sz_search, size(X, nd), size(X, nd)]);
        lambda = 0.01;  % for the rank-1 covariance case (if size(Xtmp, 1) == 1) we need to set lambda manually
        % loop and build covariance
        for s = 1:n_samples
            for ix = dim_loop
                % extract covariance data for target indices only
                Xtmp = squeeze1(X(s,ix{:},pparam.target_indices{:},:,:,:,:,:,:,:,:));

                if ismatrix(Xtmp) && size(Xtmp, 1) == 1
                    % rank 1 covariance is given by the outer vector
                    % product
                    Xtmp = Xtmp - mean(Xtmp);
                    C = Xtmp' * Xtmp;
                else
                    % Regularization
                    if ischar(pparam.lambda) && strcmp(pparam.lambda,'auto')
                        lambda = LedoitWolfEstimate(Xtmp, 'primal');
                    end
                    C = cov(Xtmp);
                end
                C = (1-lambda) * C + lambda * eye(size(C,1)) * trace(C)/size(C,1);

                % Inverse square root
                C_invsqrt(ix{:}, :, :) = squeeze(C_invsqrt(ix{:}, :, :)) + inv(sqrtm(C));
            end
        end
        C_invsqrt = C_invsqrt / n_samples;
    end

    % Save inverse squareroot of covariance
    pparam.C_invsqrt = C_invsqrt;
end

% Apply MMN
if nd == 2
    X = X * pparam.C_invsqrt;
else
    for s = 1:n_samples
        for ix = dim_loop
            X(s,ix{:},:,:,:,:,:,:,:,:,:) = squeeze1(X(s,ix{:},:,:,:,:,:,:,:,:,:)) * squeeze(pparam.C_invsqrt(ix{:}, :, :));
        end
    end

    X = permute(X, new_dim_order);

end

