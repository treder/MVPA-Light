function [pparam, X, clabel] = mv_preprocess_replacenan(pparam, X, clabel)
% Replaces sample-specific nans in the data.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_oversample(pparam, X, clabel)
%
%Parameters:
% X              - [samples x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% pparam         - [struct] with preprocessing parameters
% .sample_dimension - which dimension(s) of the data matrix represent the samples
%                     (default 1), currently only singleton sample_dimensions are supported
%
% Note: If you use replacenan with cross-validation, the replacement
% needs to be done *within* each training fold. The reason is that if the
% replacement is done globally (before starting cross-validation), the
% training and test sets are not independent any more because they might
% contain identical samples (a sample could be in the training data and its
% copy in the test data). This will make life easier for the classifier and
% will lead to an artificially inflated performance.

if numel(pparam.sample_dimension) > 1
  error('mv_preprocess_replacenan currently only supports a singleton sample_dimension');
end

if pparam.is_train_set
    
    nclasses = max(clabel);
    
    % Sample count for each class
    N = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

    sz = [size(X) 1];
    
    % reshape into a samples by features matrix for the given class
    if ~pparam.sample_dimension==1
        permutevec = [pparam.sample_dimension setdiff(1:numel(sz), pparam.sample_dimension)];
        X = permute(X, permutevec);
    end
    sz = [size(X) 1];

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
    
    if exist('permutevec', 'var')
        X = ipermute(reshape(X, sz), permutevec);
    end
end

