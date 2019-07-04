function [preprocess_param, X, clabel] = mv_preprocess_demean(preprocess_param, X, clabel)
% Demeans the data by subtracting the mean across samples.
%
%Usage:
% [preprocess_param, X, clabel] = mv_preprocess_demean(preprocess_param, X, clabel)
%
%Parameters:
% X              - [... x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% preprocess_param - [struct] with preprocessing parameters
% .dimension        - which dimension(s) of the data matrix will be used
%                     for demeaning. Typically the dimension representing
%                     the samples (default 1)
%
% Nested preprocessing: For train data, preprocess_param.mean is calculated
% and applied to the data. For test data (is_train_set = 0), the
% preprocess_param.mean obtained from the train data is used to demean the
% test data.

if preprocess_param.is_train_set
    preprocess_param.mean = mean(X, preprocess_param.dimension);
end

% Remove mean from data
X = bsxfun(@minus, X, preprocess_param.mean);

