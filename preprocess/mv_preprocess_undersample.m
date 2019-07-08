function [pparam, X, clabel] = mv_preprocess_undersample(pparam, X, clabel)
% Undersamples the majority class(es) in unbalanced data.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_undersample(pparam, X, clabel)
%
%Parameters:
% X              - [samples x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% pparam         - [struct] with preprocessing parameters
% .sample_dimension - which dimension(s) of the data matrix represent the samples
%                     (default 1)
% .undersample_test_set - by default, if undersampling is used during
%                         cross-validation, only the train set is
%                         undersampled (default 0). If the test data has to
%                         be undersampled, too, set to 1.
%
% Undersampling can safely be performed globally (on the full dataset)
% since it does not introduce any dependencies between samples.

if pparam.is_train_set || pparam.undersample_test_set
    
    sd = sort(pparam.sample_dimension(:))';
    nclasses = max(clabel);
    
    % Sample count for each class
    N = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

    % there can be multiple sample dimensions. Therefore, we build a colon
    % operator to extract the train/test samples irrespective of the
    % position and number of sample dimensions
    s = repmat({':'},[1, ndims(X)]);
    
    % undersample the majority class(es)
    rm_samples = abs(N - min(N));
    for cc=1:nclasses
        if rm_samples(cc)>0
            ix_this_class = find(clabel == cc);
            ix_rm = randperm( numel(ix_this_class), rm_samples(cc));

            % Remove samples from all sample dimensions
            for rm_dim=sd
                s_dim = s;
                s_dim(rm_dim) = {ix_this_class(ix_rm)};
                X(s_dim{:})= [];
            end
            clabel(ix_this_class(ix_rm))= [];
        end
    end
end

