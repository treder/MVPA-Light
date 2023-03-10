function [pparam, X, clabel] = mv_preprocess_impute_nan(pparam, X, clabel)
% Imputes nans in the data, so that downstream train functions return numeric results.
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
%                  electrodes x time] data. If imputation_dimension = 3 imputation is performed
%                  across time, that is, other time points (within a given trial and electrode)
%                  would be used to replace the nans.
% .method        - can be 'forward': earlier elements in the array are used e.g. [4, 6, NaN, NaN 2, 1]
%                  becomes [4, 6, 6, 6, 2, 1]),
%                  'backward': [4, 6, NaN, NaN, 2, 1] becomes [4, 6, 2, 2, 2, 1],
%                  'nearest':  [4, 6, NaN, NaN, 2, 1] becomes [4, 6, 6, 2, 2, 1],
%                  'random':  replace missing values by randomly drawing from non-NaN data
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

nan_ix = find(~isfinite(X));

if isempty(nan_ix)
    % nothing to do...
    return;
end

pparam.impute_dimension = pparam.impute_dimension(:)'; % make sure it's a row vector
sz = size(X);

% convert linear indices to row/col/... 
sub = cell(ndims(X),1);
[sub{:}] = ind2sub(sz, nan_ix);

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

    if pparam.is_train_set
        
        nclasses = max(clabel);
        
        
        sz = [size(X) 1];
        
        % reshape into a samples by features matrix for the given class -> this
        % is not needed because the caller function already has permuted the
        % data such to have the sample_dimension to be 1. Additionally, the
        % pparma.sample_dimension will have been adjusted to reflect the
        % permutation of the data matrices, which essentially renders the parameter
        % inconsistent with the data representation at this stage.
        % For now I assume the samples always to be along the first dimension.
        %if pparam.sample_dimension~=1
        %    permutevec = [pparam.sample_dimension setdiff(1:numel(sz), pparam.sample_dimension)];
        %    X = permute(X, permutevec);
        %end
        %sz = [size(X) 1];
    
        for cc=1:nclasses
            
            ix_this_class = find(clabel == cc);
            
            Xtmp = X(ix_this_class, :);
            
            % check that all features for a given sample are non-finite
            Nonfinite  = ~isfinite(Xtmp);
            Nnonfinite = sum(Nonfinite, 1);
            fprintf('Replacing on average %3.1f NaN samples (%d - %d) with surrogate data for class %d\n', mean(Nnonfinite(:)), min(Nnonfinite(:)), max(Nnonfinite(:)), cc);
            for i=1:size(Xtmp,2)
                Nonfin = Nonfinite(:,i);
                if sum(Nonfin) >= sum(~Nonfin)
                    error('There should be more samples without NaNs than samples with Nans');
                end
                selok = find(~Nonfin);
                selok = selok(randperm(sum(~Nonfin), sum(Nonfin)));
                Xtmp(Nonfin, i) = Xtmp(selok, i);
            end
            X(ix_this_class, :) = Xtmp;
            
        end
        
        % this is not needed since the permutation has been commented out above
        if exist('permutevec', 'var')
            X = ipermute(reshape(X, sz), permutevec);
        end
    end
end
