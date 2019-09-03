function [pparam, X, clabel] = mv_preprocess_demean(pparam, X, clabel)
% Demeans the data by subtracting the mean across samples.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_demean(pparam, X, clabel)
%
%Parameters:
% X              - [... x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% pparam         - [struct] with preprocessing parameters
% .dimension        - which dimension(s) of the data matrix will be used
%                     for demeaning. Typically the dimension representing
%                     the samples (default 1)
%
% Nested preprocessing: For train data, preprocess_param.mean is calculated
% and applied to the data. For test data (is_train_set = 0), the
% preprocess_param.mean obtained from the train data is used to demean the
% test data.

if pparam.is_train_set
    pparam.mean = mean(X, pparam.dimension);
end

% Remove mean from data
X = bsxfun(@minus, X, pparam.mean);

