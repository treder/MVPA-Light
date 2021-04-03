function mv_print_regression_info(cfg, X, Y, X2, Y2)
% Prints information regarding the regression procedure, including the
% cross-validation procedure and the dimensions of the input data. This
% function is called by mv_regress.
%
% Usage:
% mv_print_regression_info(cfg, <X, Y, X2, Y2>)
%
%Parameters:
% cfg       - configuration struct with parameters .model, .k, .cv,
%             and .repeat
% X         - [... x ... x ... x] dataset (optional)
% Y         - [samples x ...] responses  (optional)
% X2        - [... x ... x ... x] second dataset (optional)
% Y2        - [samples x ...] second responses dataset (optional)

if nargin <= 3 || (isempty(X2) && isempty(Y2))
    % Print type of regression
    if ~strcmp(cfg.cv,'none')
        if strcmp(cfg.cv,'kfold'), k=sprintf(' (k=%d)', cfg.k); else k=''; end
        fprintf('Performing %s cross-validation%s with %d repetitions using a %s model.\n', ...
            cfg.cv, k, cfg.repeat, upper(cfg.model))
    else
        fprintf('Training and testing on the same data using a %s model.\n',upper(cfg.model))
    end
    
    % Print data information
    if nargin>1 
        if isempty(cfg.dimension_names)
            fprintf('Data is %s.\n', strjoin(cellfun(@num2str, num2cell(size(X)),'Un',0),' x ') )
        else
            fprintf('Data is %s.\n', strjoin(cellfun(@(x,y) [num2str(x) ' ' y], num2cell(size(X)), cfg.dimension_names,'Un',0),' x ') )
        end
    end
    
    % Print response information
    if nargin>2
        fprintf('Response values: ')
        fprintf('range = [%2.5f, %2.5f], mean = %2.5f, median = %2.5f, sd = %2.5f\n', min(Y(:)), max(Y(:)), mean(Y(:)), median(Y(:)), std(Y(:)))
    end
else
    %% Transfer learning (cross regression) using two datasets
    fprintf('Performing cross regression (train on dataset 1, test on dataset 2) using a %s model.\n',upper(cfg.model));

    % Train data: Print data information
    if isempty(cfg.dimension_names)
        fprintf('Train data is %s.\n', strjoin(cellfun(@num2str, num2cell(size(X)),'Un',0),' x ') )
    else
        fprintf('Train data is %s.\n', strjoin(cellfun(@(x,y) [num2str(x) ' ' y], num2cell(size(X)), cfg.dimension_names,'Un',0),' x ') )
    end
    
    % Train data: Print response information
    fprintf('Train response values: ')
    fprintf('range = [%2.5f, %2.5f], mean = %2.5f, median = %2.5f, sd = %2.5f\n', min(Y(:)), max(Y(:)), mean(Y(:)), median(Y(:)), std(Y(:)))
    
    % Test data: Print data information
    if isempty(cfg.dimension_names)
        fprintf('Test data is %s.\n', strjoin(cellfun(@num2str, num2cell(size(X2)),'Un',0),' x ') )
    else
        fprintf('Test data is %s.\n', strjoin(cellfun(@(x,y) [num2str(x) ' ' y], num2cell(size(X2)), cfg.dimension_names,'Un',0),' x ') )
    end
    
    % Test data: Print class label information
    fprintf('Test response values: ')
    fprintf('range = [%2.5f, %2.5f], mean = %2.5f, median = %2.5f, sd = %2.5f\n', min(Y2(:)), max(Y2(:)), mean(Y2(:)), median(Y2(:)), std(Y2(:)))


end

%% Print preprocessing pipeline
if isfield(cfg, 'preprocess') && ~isempty(cfg.preprocess)
    preprocess = cellfun(@(pp) char(pp), cfg.preprocess, 'Un', 0);
    fprintf('Preprocessing: %s\n', strjoin(preprocess, ' -> '))
end

