function result = mv_prepare_plot(result, varargin)
% Adds a .plot substruct to the result structure with detailed plotting
% instructions for mv_plot_result (eg type of plot, labels for the axes).
%
%Usage:
% result = mv_prepare_plot(result, <x, y>)
%
%Parameters:
% result            - results struct obtained from one of the
%                     classification functions above. 
%
%Returns:
% result            - struct enhanced with a result.plot struct with
%                     plotting details

% (c) matthias treder

tmp = [];
tmp.metric                  = result.metric;
tmp.perf                    = result.perf;
tmp.perf_std                = result.perf_std;
tmp.perf_dimension_names    = result.perf_dimension_names;
n_metrics                   = result.n_metrics;

if n_metrics == 1
    tmp.metric   = {tmp.metric};
    tmp.perf     = {tmp.perf};
    tmp.perf_std = {tmp.perf_std};
    tmp.perf_dimension_names = {tmp.perf_dimension_names};
end

plt = cell(n_metrics);
if strcmp(result.task,'classification')
    class_labels = strcat({'Class ' }, arrayfun(@(x) {num2str(x)}, 1:result.n_classes));
end

for mm = 1:n_metrics
    
    metric      = tmp.metric{mm};
    perf        = tmp.perf{mm};
    perf_dimension_names = tmp.perf_dimension_names{mm};
    if ~iscell(perf_dimension_names), perf_dimension_names={perf_dimension_names}; end
    sz = size(perf);
    % Data dimensions (excluding 'metric' in case it is multi-dimensional like dval and confusion)
    result_dimensions = find(~ismember(perf_dimension_names,'metric'));
    n_result_dimensions = numel(result_dimensions);
    % Metric dimension (for multi-class or dval output it's multivariate)
    metric_ix = find(ismember(perf_dimension_names,'metric'));
    if isempty(metric_ix)
        size_metric_dimension = 1;
    else
        size_metric_dimension = sz(metric_ix);
    end
    
    % initialize plot fields
    p = [];
    p.title = '';
    p.combined = false;
    p.result_dimensions = result_dimensions;
    p.n_result_dimensions = n_result_dimensions;

    if strcmp(metric,'confusion')     %%% --- for CONFUSION MATRIX ---
        if n_result_dimensions == 0
            p.plot_type = 'confusion_matrix';
        else
%             p.plot_type = 'interactive';
            error('Cannot plot data that consists of confusion matrices along %d dimensions. Try to reduce the data\n', n_result_dimensions)
        end
        p.xlabel = 'Predicted class';
        p.ylabel = 'True class';
        p.title  = 'Confusion matrix';
    elseif strcmp(metric,'none')   %%% --- for NONE/RAW output ---
        error('TODO')
        
        %% HIER WEITER MACHEN %%
        
%         if n_result_dimensions
        error('Cannot plot a result that has %d data dimensions. Try to reduce the data to <= 2 dimensions\n', n_result_dimensions)

    else
        switch(n_result_dimensions)
            case 0      %%% --- for BAR PLOT ---
                p.plot_type = 'bar';
                p.ylabel    = metric;
                p.n_bars    = numel(perf);
                if strcmp(result.task,'classification') && (p.n_bars == result.n_classes)
                    p.xticklabel = class_labels;
                else
                    p.xticklabel = '';
                end
                
            case 1      %%% --- for LINE PLOT ---
                p.plot_type     = 'line';
                p.xlabel        = perf_dimension_names{1};
                p.ylabel        = metric;
                p.add_legend    = size_metric_dimension>1;
                if strcmp(result.task,'classification')
                    p.legend_labels = class_labels;
                end
                
            case 2      %%% --- for IMAGE plot ---
                p.plot_type = 'image';
                p.size_metric_dimension = size_metric_dimension;
                p.metric_ix     = metric_ix;
                if strcmp(result.task,'classification') && size_metric_dimension>1
                    p.title     = class_labels;
                end
                p.colorbar_location = 'EastOutside';
                p.global_clim       = 1;
                
                data_ix = find(~ismember(perf_dimension_names, 'metric'));
                p.plot_dimensions = data_ix([1,2]);
%                 p.interactive   = n_result_dimensions>2;
                p.ylabel        = perf_dimension_names{p.plot_dimensions(1)};
                p.xlabel        = perf_dimension_names{p.plot_dimensions(2)};
            
            otherwise   %% too high-dimensional
                error('Cannot plot a result that has %d data dimensions. Try to reduce the data to <= 2 dimensions\n', n_result_dimensions)

        end
    end
    
    % Add options for graphical elements
    p.text_options = {'Fontsize',15,'HorizontalAlignment','center'};
    p.legend_options = {'Interpreter','none'};
    p.label_options = {'Fontsize', 14, 'Interpreter', 'none'};
    p.title_options = {'Fontsize', 16, 'Fontweight', 'bold', 'Interpreter', 'none'};
    p.errorbar_options = {'Color' 'k' 'LineWidth' 2 'LineStyle' 'none'};
    
    % metric-specific settings
    switch(metric)
        case {'auc', 'acc','accuracy','precision','recall','f1'}
            p.hor = 1 / result.n_classes;
        otherwise
            p.hor = 0;     
    end
    p.climzero = p.hor;
    
    % current plot
    p.warning = '';
    
  
    plt{mm} = p;
end

result.plot = plt;