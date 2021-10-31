function [cfg, Xtrain, Ytrain, Xtest, Ytest] = mv_select_train_and_test_data(cfg, X, Y, train_indices, test_indices, is_kernel_matrix)
% Splits the data into training and test set. 
% Also selects data provided as preprocessing parameters if the .data field
% in cfg.preprocess_param is not empty (e.g. .signal and .noise in 
% mv_preprocess_ssd).
%
%Usage:
%[Xtrain, Ytrain, Xtest, Ytest] = 
%    mv_select_train_and_test_data(cfg, X, Y, train_indices, test_indices, is_kernel_matrix)
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

% Also select preprocessing data
for p=1:numel(cfg.preprocess_fun)
    for fn = cfg.preprocess_param{p}.select_data
        cfg.preprocess_param{p}.([fn{1} '_train']) = cfg.preprocess_param{p}.(fn{1})(train_indices,:,:,:);
        cfg.preprocess_param{p}.([fn{1} '_test']) = cfg.preprocess_param{p}.(fn{1})(test_indices,:,:,:);
    end
end
