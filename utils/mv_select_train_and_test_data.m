function [Xtrain, Ytrain, Xtest, Ytest] = mv_select_train_and_test_data(X, Y, train_indices, test_indices, is_kernel_matrix)
%Splits the data into training and test set. 
%
%Usage:
%[Xtrain, Ytrain, Xtest, Ytest] = 
%    mv_select_train_and_test_data(X, Y, train_indices, test_indices, is_kernel_matrix)
%
%Parameters:
% X              - [samples x ... x ... ] data matrix -OR-
%                  [samples x samples  x ... ] kernel matrix
% Y              - [samples x ...] vector of class labels or vector/matrix
%                                  of responses
% train_indices  - vector of indices of train samples
% test_indices   - vector of indices of test samples
% is_kernel_matrix - if 1, X represents a kernel matrix
%
%Returns:
% Xtrain, Ytrain     - train data and class labels/responses
% Xtest, Ytest       - test data and class labels/responses

if ~is_kernel_matrix
    % Standard case: samples are in the first dimension only
    
    % Get train and test data
    Xtrain = X(train_indices,:,:,:,:,:,:,:,:,:,:,:,:,:,:);
    Xtest= X(test_indices,:,:,:,:,:,:,:,:,:,:,:,:,:,:);
    
else
    % kernel matrix is provided: we need to select a 
    % [train samples x train samples] matrix for training and a 
    % [test samples x train samples]  matrix for testing
    Xtrain = X(train_indices, train_indices,:,:,:,:,:,:,:,:,:,:,:,:,:);
    Xtest= X(test_indices, train_indices,:,:,:,:,:,:,:,:,:,:,:,:,:,:);
    
%     % Non-standard case: sample dimension is not 1, or there is multiple
%     % sample dimensions.
%     % In this case, we will build a colon operator s to extract
%     % the train/test samples
%     s_train = repmat({':'},[1, ndims(X)]);
%     s_test  = repmat({':'},[1, ndims(X)]);
%     
%     s_train(cfg.sample_dimension) = {train_indices};
%     s_test(cfg.sample_dimension) = {test_indices};
%     
%     % Get train and test data
%     Xtrain = X(s_train{:});
%     Xtest= X(s_test{:});
end

% Get train and test class labels
Ytrain = Y(train_indices,:);
Ytest  = Y(test_indices,:);
