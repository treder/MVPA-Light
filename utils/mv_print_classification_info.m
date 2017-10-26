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
% cfg       - configuration struct with parameters .classifier, .K, .CV,
%             and .repeat
% X         - [samples x features] struct with data (optional)
% clabel    - [samples x 1] class labels  (optional)
% X2        - [samples x features] struct with second dataset (optional)
% clabel2   - [samples x 1] class labels for second dataset (optional)

if nargin <= 3
    % Print type of classification
    if ~strcmp(cfg.CV,'none')
        fprintf('Performing %s cross-validation (K=%d) with %d repetitions using a %s classifier.\n', ...
            cfg.CV, cfg.K, cfg.repeat, upper(cfg.classifier))
    else
        fprintf('Training and testing on the same data using a %s classifier.\n',upper(cfg.classifier))
    end
    
    % Print data information
    if nargin>1
        if ndims(X)==2
            fprintf('Data has %d samples and %d features.\n', size(X))
        elseif ndims(X)==3
            fprintf('Data has %d samples, %d features, and %d time points.\n', size(X))
        end
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

    if ndims(X)==2
        fprintf('Dataset 1 has %d samples and %d features.\n', size(X))
    elseif ndims(X)==3
        fprintf('Dataset 1 has %d samples, %d features, and %d time points.\n', size(X))
    end
    
    
    u = unique(clabel);
    fprintf('Class frequencies: ');
    for ii=1:numel(u)
        fprintf('%2.2f%% [class %d]', 100*sum(clabel==u(ii))/numel(clabel), u(ii));
        if ii<numel(u), fprintf(', '), end
    end
    fprintf('.\n')
    
    if ndims(X)==2
        fprintf('Dataset 2 has %d samples and %d features.\n', size(X2))
    elseif ndims(X)==3
        fprintf('Dataset 2 has %d samples, %d features, and %d time points.\n', size(X2))
    end
    
    
    u = unique(clabel2);
    fprintf('Class frequencies (dataset 2): ');
    for ii=1:numel(u)
        fprintf('%2.2f%% [class %d]', 100*sum(clabel2==u(ii))/numel(clabel2), u(ii));
        if ii<numel(u), fprintf(', '), end
    end
    fprintf('.\n')
end
