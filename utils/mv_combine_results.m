function result = mv_combine_results(results, combine)
% Combines multiple result structs for plotting. This is useful for
% plotting eg multiple subjects or classifiers.
%
% The can only be combined if all results were obtained on data of the same 
% size using the same metrics.
%
%Usage:
% result = mv_combine_results(results, approach)
%
%Parameters:
% results           - cell array of result structs
% combine           - specifies how the results are combined
%                     'merge' merges the results into one plot (default).
%                             This works for bar plots (leading to grouped
%                             bars) and line plots (multiple lines). It
%                             does not work for other types of plots.
%                     'average' calculates the average values across all
%                             result structs and stores them in perf.
%                             perf_std is recalculated as the std deviation
%                             across results.
%
%Returns:
% result            - struct enhanced with a result.plot struct with
%                     plotting details

% (c) matthias treder

if ~iscell(results)
    error('Expecting a cell array of results as input'); 
elseif numel(results)<2
    error('Expecting at least 2 result structs'); 
end
if nargin<2, combine = 'merge'; end

n_results   = numel(results);
n_metrics   = results{1}.n_metrics;

% make sure all results have a .plot field
for nn=1:n_results
    if ~isfield(results{nn},'plot')
        results{nn} = mv_prepare_plot(results{nn});
    end
end

% make sure perf and perf_std are cell arrays
for nn=1:n_results
    if ~iscell(results{nn}.perf), results{nn}.perf = {results{nn}.perf}; end
    if ~iscell(results{nn}.perf_std), results{nn}.perf_std = {results{nn}.perf_std}; end
    if ~iscell(results{nn}.perf_dimension_names), results{nn}.perf_dimension_names = {results{nn}.perf_dimension_names}; end
end

% if the results have no .name fields, initialize them by the
% classifier/model's name
for nn=1:n_results
    if ~isfield(results{nn},'name')
        if isfield(results{nn},'classifier')
            results{nn}.name = results{nn}.classifier;
        else
            results{nn}.name = results{nn}.model;
        end
    end
end

name_labels = cellfun(@(res) res.name, results, 'Un', 0);

% combine results
result = results{1};
rm_idx = [];  % store indices of plots that need to be removed
for mm=1:n_metrics
    if strcmp(combine,'merge')
        % only bar plots and line plots can be combined
        if strcmp(results{1}.plot{mm}.plot_type, 'bar')
            result.plot{mm}.combined = 1;
            c = cellfun( @(res) res.perf{mm}, results, 'Un', 0);
            result.perf{mm} = cat(2, c{:});
            c = cellfun( @(res) res.perf_std{mm}, results, 'Un', 0);
            result.perf_std{mm} = cat(2, c{:});
            if isvector(result.perf{mm})
                % if the metric is univariate, we will get a separate bar
                % for each result
                result.plot{mm}.n_bars      = n_results;
                result.plot{mm}.xticklabel  = name_labels;
            else
                % if the metric is multivariate, we will get a grouped bar
                % plot (where each bar represents one class)
                result.plot{mm}.legend_labels = name_labels;
            end

        elseif strcmp(results{1}.plot{mm}.plot_type, 'line')
            result.plot{mm}.combined = 1;
            if isvector(results{1}.perf{mm})
                catdim = 2;
                % different results will be multiple lines
                result.plot{mm}.add_legend    = true;
                result.plot{mm}.legend_labels = name_labels;
            else
                % each result is already multivariate (eg dvals) - we
                % append results in the 3rd dimension and then generate
                % separate plots
                catdim = 3;
                result.plot{mm}.title = name_labels;
            end
            c = cellfun( @(res) res.perf{mm}, results, 'Un', 0);
            result.perf{mm} = cat(catdim, c{:});
            c = cellfun( @(res) res.perf_std{mm}, results, 'Un', 0);
            result.perf_std{mm} = cat(catdim, c{:});
        else
            warning('Cannot merge plots of type ''%s'', skipping...', results{1}.plot{mm}.plot_type)
            rm_idx = [rm_idx mm];
        end
        
    elseif strcmp(combine,'average')
        % find last non-singleton index 
        ix = find(size(results{1}.perf{mm})>1,1,'last');
        c = cellfun( @(res) res.perf{mm}, results, 'Un', 0);
        c = cat(ix+1, c{:}); % append along new dimension
        result.perf{mm} = mean(c, ix+1);
        result.perf_std{mm} = std(c, [], ix+1);
        result.plot{mm}.title = 'average';
    else
        error('Unknown approach: %s', combine)
    end
end

% remove results that could not be merged
if ~isempty(rm_idx)
    result.perf(rm_idx) = [];
    result.perf_std(rm_idx) = [];
    result.perf_dimension_names(rm_idx) = [];
    result.plot(rm_idx) = [];
end

if n_metrics == 1 && ~isempty(result.perf)
    result.perf = result.perf{1};
    result.perf_std = result.perf_std{1};
    result.perf_dimension_names = result.perf_dimension_names{1};
end