function [preprocess_param, X, clabel] = mv_preprocess_zscore(preprocess_param, X, clabel)
% Z-scores the data to have mean=0 and std=1. It does so by subtracting the 
% mean across samples and dividing by the standard deviation. Z-scoring is 
% always performed across the first dimension, for each of the other
% dimensions separately.
%
%Usage:
% [preprocess_param, X, clabel] = mv_preprocess_zscore(preprocess_param, X, clabel)
%
%Parameters:
% X              - [... x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% preprocess_param - [struct] with preprocessing parameters
%                    function (not required here since zscore has not parameters)
% .dimension        - which dimension(s) of the data matrix will be used
%                     for z-scoring. Typically it should be the dimension
%                     representing the samples (default 1)
%
% Nested preprocessing: For train data, preprocess_param.mean and 
% preprocess_param.standard_deviation are calculated
% and applied to the data. For test data (is_train_set = 0), both parameters 
% obtained from the train data are used to scale the test data.

if preprocess_param.is_train_set

    % calculate mean and standard deviation on the (train) data
    preprocess_param.mean = mean(X, preprocess_param.dimension);
    preprocess_param.standard_deviation = std(X, preprocess_param.dimension);
    
end

% Set array size for repmat
rep = ones(1, ndims(X));
rep(preprocess_param.dimension) = size(X, preprocess_param.dimension);

% Remove mean from data and divide by standard deviation
X = X - repmat(preprocess_param.mean, rep);
X = X ./ repmat(preprocess_param.standard_deviation, rep);

