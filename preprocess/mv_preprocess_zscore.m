function [pparam, X, clabel] = mv_preprocess_zscore(pparam, X, clabel)
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
% pparam         - [struct] with preprocessing parameters
% .dimension        - which dimension(s) of the data matrix will be used
%                     for z-scoring. Typically the dimension representing
%                     the samples (default 1)
%
% Nested preprocessing: For train data, preprocess_param.mean and 
% preprocess_param.standard_deviation are calculated
% and applied to the data. For test data (is_train_set = 0), both parameters 
% obtained from the train data are used to scale the test data.

if pparam.is_train_set
    pparam.mean = mean(X, pparam.dimension);
    pparam.standard_deviation = std(X, [], pparam.dimension);    
end

% Remove mean from data and divide by standard deviation
X = bsxfun(@minus, X, pparam.mean);
X = bsxfun(@(x,y) x ./ y, X, pparam.standard_deviation);

