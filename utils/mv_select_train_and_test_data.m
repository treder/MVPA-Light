function [Xtrain, trainlabel, Xtest, testlabel] = mv_select_train_and_test_data(cfg, X, clabel, train_indices, test_indices)
%Splits the data into training and test set. 
%
%Usage:
%[Xtrain, trainlabel, Xtest, testlabel] = 
%    mv_select_train_and_test_data(cfg, X, clabel, train_indices, test_indices)
%
%Parameters:
% X              - [... x ... x ... x] data matrix
% clabel         - [samples x 1] vector of class labels
% train_indices  - vector of indices of train samples
% test_indices   - vector of indices of test samples
%
% cfg          - struct with parameters:
% .sample_dimension - vector specifying which dimension of the array X
%                     contains the samples (can be multiple). Samples are
%                     selected along this dimension(s).
%
%
%Returns:
% Xtrain, trainlabel - train data and class labels
% Xtest, testlabel   - test data and class labels

if all(cfg.sample_dimension == 1)
    % Standard case: samples are in the first dimension only
    
    % Get train and test data
    Xtrain = X(train_indices,:,:,:,:,:,:,:,:,:,:,:,:);
    Xtest= X(test_indices,:,:,:,:,:,:,:,:,:,:,:,:);
    
else
    % Non-standard case: sample dimension is not 1, or there is multiple
    % sample dimensions.
    % In this case, we will build a colon operator s to extract
    % the train/test samples
    s_train = repmat({':'},[1, ndims(X)]);
    s_test  = repmat({':'},[1, ndims(X)]);
    
    s_train(cfg.sample_dimension) = {train_indices};
    s_test(cfg.sample_dimension) = {test_indices};
    
    % Get train and test data
    Xtrain = X(s_train{:});
    Xtest= X(s_test{:});
end

% Get train and test class labels
trainlabel = clabel(train_indices);
testlabel  = clabel(test_indices);
