function mv_print_classification_info(cfg, X, clabel, X2, clabel2)
% Prints information regarding the classification procedure, including the
% cross-validation procedure and the dimensions of the input data. This
% function is called by mv_crossvalidate, mv_classify_across_time, and
% mv_classify_timextime.
%
% Usage:
% mv_print_classification_info(cfg, <X, clabel, X2, clabel2>)
%
%Parameters:
% cfg       - configuration struct with parameters .classifier, .k, .cv,
%             and .repeat
% X         - [... x ... x ... x] dataset (optional)
% clabel    - [samples x 1] class labels  (optional)
% X2        - [... x ... x ... x] second dataset (optional)
% clabel2   - [samples x 1] class labels for second dataset (optional)

if nargin <= 3 || (isempty(X2) && isempty(clabel2))
    % Print type of classification
    if ~strcmp(cfg.cv,'none')
        if strcmp(cfg.cv,'kfold'), k=sprintf(' (k=%d)', cfg.k); else k=''; end
        fprintf('Performing %s cross-validation%s with %d repetitions using a %s classifier.\n', ...
            cfg.cv, k, cfg.repeat, upper(cfg.classifier))
    else
        fprintf('Training and testing on the same data using a %s classifier.\n',upper(cfg.classifier))
    end
    
    % Print data information
    nclasses = max(clabel);
    if nargin>1 
        dimensions = arrayfun(@(x,y) sprintf('%d %s, ', x, y{:}), [size(X) ones(1, length(cfg.dimension_names)-ndims(X))], cfg.dimension_names(:)', 'Un', 0);
        fprintf('Data has %sand %d classes.\n', [dimensions{:}], nclasses)
    end
    
    % Print class label information
    if nargin>2
        u = unique(clabel);
        fprintf('Class frequencies: ');
        for ii=1:numel(u)
            fprintf('%2.2f%% [class %d]', 100*sum(clabel==u(ii))/numel(clabel), u(ii));
            if ii<numel(u), fprintf(', '), end
        end
        fprintf('.\n')
    end
    
else
    %% Transfer classification (cross decoding) using two datasets
    fprintf('Performing cross-classification (train on dataset 1, test on dataset 2) using a %s classifier.\n',upper(cfg.classifier));

    % Train data: Print data information
    nclasses = max(clabel);
    if nargin>1 
        dimensions = arrayfun(@(x,y) sprintf('%d %s, ', x, y{:}), [size(X) ones(1, length(cfg.dimension_names)-ndims(X))], cfg.dimension_names(:)', 'Un', 0);
        fprintf('Train data has %sand %d classes.\n', [dimensions{:}], nclasses)
    end
    
    % Train data: Print class label information
    u = unique(clabel);
    fprintf('Train data class frequencies: ');
    for ii=1:numel(u)
        fprintf('%2.2f%% [class %d]', 100*sum(clabel==u(ii))/numel(clabel), u(ii));
        if ii<numel(u), fprintf(', '), end
    end
    fprintf('.\n')
    
    % Test data: Print data information
    nclasses = max(clabel2);
    if nargin>1 
        dimensions = arrayfun(@(x,y) sprintf('%d %s, ', x, y{:}), [size(X2) ones(1, length(cfg.dimension_names)-ndims(X))], cfg.dimension_names(:)', 'Un', 0);
        fprintf('Test data has %sand %d classes.\n', [dimensions{:}], nclasses)
    end
    
    % Test data: Print class label information
    u = unique(clabel2);
    fprintf('Test data class frequencies: ');
    for ii=1:numel(u)
        fprintf('%2.2f%% [class %d]', 100*sum(clabel2==u(ii))/numel(clabel2), u(ii));
        if ii<numel(u), fprintf(', '), end
    end
    fprintf('.\n')

end

%% Print preprocessing pipeline
if isfield(cfg, 'preprocess') && ~isempty(cfg.preprocess)
    preprocess = cellfun(@(pp) char(pp), cfg.preprocess, 'Un', 0);
    fprintf('Preprocessing: %s\n', strjoin(preprocess, ' -> '))
end

end
