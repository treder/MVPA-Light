function result = mv_select_result(result, metric)
% For a result(s) with multiple metrics, selects a metric.
%
%Usage:
% result = mv_select_result(results, metric)
%
%Parameters:
% result            - result struct or cell array of result structs
% metric            - desired metric. All other metrics are removed.
%
%Returns:
% result            - struct(s) with selected metric

% (c) matthias treder

undo_cell = 0;
if ~iscell(result)
    result = {result};
    undo_cell = 1;
end

n_results   = numel(result);

for n=1:n_results
    if ischar(result{n}.metric) && ~strcmp(result{n}.metric, metric)
        error('Metric ''%s'' not found in result struct', metric)
    else
        ix = find(ismember(result{n}.metric, metric));
        if isempty(ix)
            error('Metric ''%s'' not found in result struct', metric)
        else
            result{n}.metric    = metric;
            result{n}.perf      = result{n}.perf{ix};
            result{n}.perf_std  = result{n}.perf_std{ix};
            result{n}.perf_dimension_names = result{n}.perf_dimension_names{ix};
            if isfield(result{n},'plot')
                result{n}.plot  = result{n}.plot{ix};
            end
            if isfield(result{n},'cfg')
                result{n}.cfg.metric = metric;
            end
            result{n}.n_metrics = 1;
        end
    end
end

if undo_cell
    result = result{1};
end