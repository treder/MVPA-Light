function param = mv_tune_hyperparameters(param, X, y, train_fun, test_fun, eval_fun, tune_params, k, is_kernel_matrix)
% Generic hyperparameter tuning function using a search grid with
% cross-validation.
%
% Usage:
% ix = mv_tune_hyperparameters(param, X, y, train_fun, test_fun, tune_params)
%
%Parameters:
% param          - struct with hyperparameters for the classifier/model
% X              - [samples x features] matrix of training samples  -OR-
%                  [samples x samples] kernel matrix
% Y              - class labels or vector/matrix of regression targets
% train_fun      - training function (e.g. @train_lda)
% test_fun       - test function (e.g. @test_lda)
% eval_fun       - evaluation function that take y and predicted y as
%                  inputs and returns a metric (e.g. accuracy, MSE). Note
%                  that we are looking for MAXIMA of the evaluation
%                  function, so for error metrics such as MSE one should
%                  provide -MSE instead.
% tune_params    - cell array specifying which hyperparameters need to be tuned
% k              - number of folds for nested cross-validation
% is_kernel_matrix - indicates whether X is a kernel matrix
%
% Returns:
% param          - updated param struct with best hyperparameters selected

% Init variables
if nargin<9, is_kernel_matrix = 0; end


% CV partition object for cross-validation
CV = cvpartition(size(X,1),'KFold', k);

% Extract values for all hyperparameters that should be tuned
n_tune_fields = numel(tune_params);
tune_values = cellfun( @(f) param.(f), tune_params, 'Un', 0);
tune_indices = cellfun( @(v) arrayfun(@(x) {x}, 1:numel(v)), tune_values, 'Un', 0);

% Compute search grid by producing all combinations of hyperparameters
search_grid = allcomb(tune_indices{:});
% ixcomb = cellfun(@(x) 1:numel(x), tune_kernel_parameters(2:2:end), 'Un', 0);
% kernel_comb_ix = allcomb(ixcomb{:});

% keeps the 
eval_values = zeros(size(search_grid, 1), 1);

tmp_param = param;

% --search grid
for ix=1:size(search_grid,1)       

    % Set hyperparameters according to current iteration
    for t=1:n_tune_fields
        tmp_param.(tune_params{t}) = param.(tune_params{t})(search_grid{ix,t});
    end
    
    for f=1:k        % --- CV folds
        % Get train and test data
        [X_train, y_train, X_test, y_test] = mv_select_train_and_test_data(X, y, CV.training(f), CV.test(f), is_kernel_matrix);

        % Train model
        model = train_fun(tmp_param, X_train, y_train);
        
        % Test model
        y_pred = test_fun(model, X_test);
        
        % Get value for evaluation function
        eval_values(ix) = eval_values(ix) + eval_fun(y_test, y_pred);
    end
end

% determine best hyperparameters
[~, max_ix] = max(eval_values);

% Set best hyperparameter
for t=1:n_tune_fields
    param.(tune_params{t}) = param.(tune_params{t})(search_grid{max_ix,t});
end
