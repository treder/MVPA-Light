function [pparam, X, clabel] = mv_preprocess_oversample(pparam, X, clabel)
% Oversamples the minority class(es) in unbalanced data.
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
%                     (default 1)
% .oversample_test_set - by default, if oversampling is used during
%                        cross-validation, only the train set is
%                        oversampled. If the test data has to
%                        be oversampled, too, set to 1 (default 0).
% .replace             - for oversampling, if 1, data is oversampled with replacement (like in
%                        bootstrapping) i.e. samples from the minority class can
%                        be added 3 or more times. If 0, data is oversampled
%                        without replacement. Note: If 0, the majority class must
%                        be at most twice as large as the minority class
%                        (otherwise we run out of samples) (default 1)
%
% Note: If you use oversampling with cross-validation, the oversampling
% needs to be done *within* each training fold. The reason is that if the
% oversampling is done globally (before starting cross-validation), the
% training and test sets are not independent any more because they might
% contain identical samples (a sample could be in the training data and its
% copy in the test data). This will make life easier for the classifier and
% will lead to an artificially inflated performance.

if pparam.is_train_set || pparam.oversample_test_set
    
    sd = sort(pparam.sample_dimension(:))';
    nclasses = max(clabel);
    
    % Sample count for each class
    N = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

    % there can be multiple sample dimensions. Therefore, we build a colon
    % operator to extract the train/test samples irrespective of the
    % position and number of sample dimensions
    s = repmat({':'},[1, ndims(X)]);
    
    % oversample the minority class(es)
    add_samples = abs(N - max(N));
    for cc=1:nclasses
        if add_samples(cc)>0
            ix_this_class = find(clabel == cc);
            if pparam.replace
                ix_add = randi( numel(ix_this_class), add_samples(cc), 1);
            else
                ix_add = randperm( numel(ix_this_class), add_amples(cc));
            end

            % Add samples to all sample dimensions
            for add_dim=sd
                s_dim = s;
                s_dim(add_dim) = {ix_this_class(ix_add)};
                X= cat(add_dim, X, X(s_dim{:}));

            end
            % Add to class labels
            clabel(end+1:end+add_samples(cc))= cc;
        end
    end
end

