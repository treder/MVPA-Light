function [pparam, X, clabel] = mv_preprocess_impute_nan(pparam, X, clabel)
% Imputes nan and inf values in the data, so that downstream train functions 
% return numeric results.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_impute_nan(pparam, X, clabel)
%
%Parameters:
% X              - [samples x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% pparam         - [struct] with preprocessing parameters
% .impute_dimension -  dimension(s) along with imputation is performed. E.g. imagine [samples x 
%                  electrodes x time] data. If impute_dimension = 3 imputation is performed
%                  across time, that is, other time points (within a given trial and electrode)
%                  would be used to replace the nans.
% .method        - can be 'forward': earlier elements in the array are used e.g. [4, 6, NaN, NaN 2, 1]
%                  becomes [4, 6, 6, 6, 2, 1]),
%                  'backward': [4, 6, NaN, NaN, 2, 1] becomes [4, 6, 2, 2, 2, 1],
%                  'nearest':  [4, 6, NaN, NaN, 2, 1] becomes [4, 6, 6, 2, 2, 1],
%                  'random':  replace missing values by randomly drawing from non-NaN data
% .fill          - it is possible that after the imputation the array still
%                  contains Nans (e.g. the array [NaN, 1, 2, NaN] with
%                  'forward' impute yields [NaN, 1, 2, 2]). In this case, the
%                  leftover Nans can be filled with the fill value (default
%                  Nan, no filling is done)
%
% Further parameters for RANDOM impute:
% .use_clabel    - (used only in 'random' impute) if the impute dimension
%                  is the sample dimension, then clabels can be used to
%                  apply imputation only to samples of the same class
%                  (default 0). 
%
% FORWARD and BACKWARD impute:
% Simple imputation approach that uses the preceding or following values
% along the impute dimensions to fill missing values. The imputation can be
% along multiple dimensions. In this case, Nans are first filled along the
% first impute dimension, any remaining nans along the second impute
% dimension, etc.
%
% RANDOM impute:
% Just supports a single impute dimension. Any Nans along this dimension
% are filled in with values from a randomly chosen slice that is free of
% nans. If use_clabel=1 only slices from the same class are selected for
% the impute.
%
%
% Note: If you use impute_nan with cross-validation, the replacement
% needs to be done *within* each training fold. The reason is that if the
% replacement is done globally (before starting cross-validation), the
% training and test sets are not independent any more because they might
% contain identical samples (a sample could be in the training data and its
% copy in the test data). This will make life easier for the classifier and
% will lead to an artificially inflated performance.

% if numel(pparam.impute_dimension) > 1
%   error('mv_preprocess_replacenan currently only supports a singleton sample_dimension');
% end

pparam.impute_dimension = pparam.impute_dimension(:)'; % make sure it's a row vector
sz = size(X);

if strcmp(pparam.method, 'forward') || strcmp(pparam.method, 'backward')
    for dim = pparam.impute_dimension
        if all(isfinite(X(:)))
            break
        end
        % permute such that the current impute dimension is in the columns
        % and hence accessible via linear indexing
        X = permute(X, [dim, setdiff(1:ndims(X), dim)]);
        nan_ix = find(~isfinite(X));

        if strcmp(pparam.method, 'forward')
            % --- FORWARD IMPUTE ---
            % remove index corresponding to start of each col since 
            % forward fill doesn't work here
            nan_ix = nan_ix(mod(nan_ix, sz(dim)) ~= 1);
    
            % fill nan with previous element
            X(nan_ix) = X(nan_ix-1);
    
            % if there are nans left it means that multiple nans follow each 
            % other and we need to loop through these blocks of nans
            nan_ix = nan_ix(~isfinite(X(nan_ix)));
            if ~isempty(nan_ix)
                d = diff(nan_ix);
                % all contiguous blocks of Nans have d==1 so the breaks 
                % tell us where the contiguous blocks end
                end_indices = find(d>1);
                end_indices = [end_indices(:)' numel(nan_ix)];
                start_ix = 1;
                for end_ix = end_indices
                    X(nan_ix(start_ix:end_ix)) = X(nan_ix(start_ix)-1);
                    start_ix = end_ix+1;
                end
            end
        elseif strcmp(pparam.method, 'backward')
            % --- BACKWARD IMPUTE ---
            % remove index corresponding to end of each col since 
            % forward fill doesn't work here
            nan_ix = nan_ix(mod(nan_ix, sz(dim)) ~= 0);
    
            % fill nan with following element
            X(nan_ix) = X(nan_ix+1);
    
            % if there are nans left it means that multiple nans follow each 
            % other and we need to loop through these blocks of nans
            nan_ix = nan_ix(~isfinite(X(nan_ix)));
            if ~isempty(nan_ix)
                d = diff(nan_ix);
                % all contiguous blocks of Nans have d==1 so the breaks 
                % tell us where the contiguous blocks end
                end_indices = find(d>1);
                end_indices = [end_indices(:)' numel(nan_ix)];
                start_ix = 1;
                for end_ix = end_indices
                    X(nan_ix(start_ix:end_ix)) = X(nan_ix(end_ix)+1);
                    start_ix = end_ix+1;
                end
            end
        end

        % permute back to original dimensions
        back_to_original_dims = [2:dim 1 dim+1:ndims(X)];
        X = permute(X, back_to_original_dims);

    end
    
elseif strcmp(pparam.method, 'random')
    % --- RANDOM IMPUTE ---
    if all(isfinite(X(:))), return; end
    assert(numel(pparam.impute_dimension)==1, 'for method=''random'' only a single impute_dimension is supported')
    if pparam.use_clabel, assert(size(X,pparam.impute_dimension)==length(clabel), 'if use_clabel=1 length of clabel should match the size of impute dimension'); end
    dim = pparam.impute_dimension;
    X = permute(X, [dim, setdiff(1:ndims(X), dim)]);
    nan_sum = sum(~isfinite(X), 2:ndims(X));
    nan_ix = find(nan_sum > 0);
    non_nan_ix = find(nan_sum == 0);

    if any(non_nan_ix)

        if ~pparam.use_clabel
            random_ix = non_nan_ix(randi(numel(non_nan_ix), numel(nan_ix), 1));
        else
            random_ix = zeros(numel(nan_ix), 1);
            for cc = 1:max(clabel)
                class_ix = clabel(nan_ix)==cc;
                nan_ix_per_class = nan_ix(class_ix);
                if isempty(nan_ix_per_class)
                    continue
                end
                non_nan_ix_per_class = non_nan_ix(clabel(non_nan_ix)==cc);
                random_ix_per_class = non_nan_ix_per_class(randi(numel(non_nan_ix_per_class), numel(nan_ix_per_class), 1));
                random_ix(class_ix) = random_ix_per_class;
            end

        end

        % loop through all slices that have any nans and replace the nans
        for ix = 1:length(nan_ix)
            X(nan_ix(ix), isnan(X(nan_ix(ix), :))) = X(random_ix(ix), isnan(X(nan_ix(ix), :)));
        end
    end

    % permute back to original dimensions
    back_to_original_dims = [2:dim 1 dim+1:ndims(X)];
    X = permute(X, back_to_original_dims);

end

% fill any leftover nans
if ~isnan(pparam.fill)
    X(~isfinite(X)) = pparam.fill;
end
