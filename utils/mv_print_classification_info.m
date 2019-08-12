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

if nargin <= 3
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
        dimensions = arrayfun(@(x,y) sprintf('%d %s, ', x, y{:}), size(X), cfg.dimension_names(:)', 'Un', 0);
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
    %% Transfer classification using two datasets
    fprintf('Training on dataset 1, testing on dataset 2 using a %s classifier.\n',upper(cfg.classifier));

    % Dataset 1
    nclasses1 = max(clabel);
    if ndims(X)==2
        fprintf('Dataset 1 has %d samples, %d features and %d classes.\n', [size(X), nclasses1])
    elseif ndims(X)==3
        fprintf('Dataset 1 has %d samples, %d features, %d time points, and %d classes.\n', [size(X), nclasses1])
    end

    u = unique(clabel);
    fprintf('Class frequencies: ');
    for ii=1:numel(u)
        fprintf('%2.2f%% [class %d]', 100*sum(clabel==u(ii))/numel(clabel), u(ii));
        if ii<numel(u), fprintf(', '), end
    end
    fprintf('.\n')
    
    % Dataset 2
    nclasses2 = max(clabel2);
    if nargin>1
        dimensions = arrayfun(@(x,y) sprintf('%d %s, ', x, y{:}), size(X2), cfg.dimension_names(:)', 'Un', 0);
        fprintf('Dataset 2 has %sand %d classes.\n', [dimensions{:}], nclasses2)
    end

    u = unique(clabel2);
    fprintf('Class frequencies (dataset 2): ');
    for ii=1:numel(u)
        fprintf('%2.2f%% [class %d]', 100*sum(clabel2==u(ii))/numel(clabel2), u(ii));
        if ii<numel(u), fprintf(', '), end
    end
    fprintf('.\n')

end

end
