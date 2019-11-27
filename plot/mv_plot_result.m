function h = mv_plot_result(result, varargin)
%Provides a simple visual representation of the results obtained with the
%functions mv_crossvalidate, mv_classify_across_time, mv_classify_timextime, 
%mv_searchlight, mv_classify, and mv_regress.
%
%The type of plot depends on which of these functions was used and on the
%dimensionality of the data.
%
%Usage:
% h = mv_plot_result(result,<...>)
%
%Parameters:
% result            - results struct obtained from one of the
%                     classification functions above. A cell array of
%                     results can be provided (e.g. results for different
%                     subjects); in this case, all results need to be 
%                     created with the same function using the same metric.
%                     If multiple metrics have been used, a separate plot
%                     is generated for each metric.
%                     
%
% ADDITIONAL KEY-VALUE ARGUMENTS:
% new_figure     - if 1, results are plotted in a new figure. If 0, results
%                  are plotted in the current axes instead (default 1)
% combine        - if multiple results are given, specifies how to combined
%                  them (see mv_combine_results for details)
% mask           - binary mask (eg statistical significance). Masks out
%                  data in images. For lines, produces bold lines.
%
% RETURNS:
% h        - struct with handles to the graphical elements 

% (c) matthias treder

% Parse any key-value pairs
opt = mv_parse_key_value_pairs(varargin{:});
if ~isfield(opt,'new_figure'), opt.new_figure = 1; end
if ~isfield(opt,'combine'), opt.combine = 'merge'; end
if ~isfield(opt,'mask'), opt.mask = []; end

% if multiple results are provided, they must be combined first
if iscell(result) && numel(result)>1
    result = mv_combine_results(result, opt.combine);
    if isempty(result.perf), error('Could not combine results'), end
end

% prepare plot if necessary
if ~isfield(result, 'plot')
    result = mv_prepare_plot(result);
end

n_metrics               = result.n_metrics;

if n_metrics == 1
    result.metric   = {result.metric};
    result.perf     = {result.perf};
    result.perf_std = {result.perf_std};
    result.perf_dimension_names = {result.perf_dimension_names};
end


for mm=1:n_metrics
    
    if opt.new_figure, figure, else, clf; end

    p = result.plot{mm};
    metric                  = result.metric{mm};
    perf                    = result.perf{mm};
    perf_std                = result.perf_std{mm};
    
    switch(p.plot_type)
        
        case 'confusion_matrix'     
            % ---------- CONFUSION MATRIX ----------
            n_classes = result.n_classes;
            imagesc(perf)
            colorbar
            h.xlabel(mm) = xlabel(p.xlabel, p.label_options{:});
            h.ylabel(mm) = ylabel(p.ylabel, p.label_options{:});
            set(gca,'Xtick',1:n_classes,'Ytick',1:n_classes)
            for rr=1:n_classes
                for cc=1:n_classes
                    if perf(rr,cc) < 0.005 
                        % if it would appear as a "0.00" we just plot a "0"
                        text(cc,rr, '0', p.text_options{:})
                    else
                        text(cc,rr, sprintf('%0.2f',perf(rr,cc)), p.text_options{:})
                    end
                end
            end
            h.title(mm) = title(p.title, p.title_options{:});

        case 'bar'                  
            % ---------- BAR PLOT ----------
            n_bars = p.n_bars;

            h.bar = bar(1:n_bars, perf);
            hold on
            if p.combined   % multiple results combined
                if ~isvector(perf)
                    % grouped bar graph
                    set(gcf,'defaultLegendAutoUpdate','off');
                    legend(result.plot{mm}.legend_labels);
                    % find centers of grouped bars
                    offset = [h.bar.XOffset];
                    xd = h.bar(1).XData;
                    centers = zeros(size(perf));
                    for bb = 1:n_bars
                        centers(bb,:) = offset + xd(bb);
                    end
                    % need to place the errorbars over the grouped bars now
                    errorbar(centers(:), perf(:), perf_std(:), p.errorbar_options{:})
                else
                    errorbar(1:n_bars, perf, perf_std, p.errorbar_options{:})
                end
            else
                % Indicate SEM if the bars are not grouped
                errorbar(1:n_bars, perf, perf_std, p.errorbar_options{:})
            end
            set(gca,'XTick',1:n_bars, 'XTickLabel', p.xticklabel)
            h.ylabel(mm) = ylabel(p.ylabel, p.label_options{:});
            h.title     = title(p.title);
            
        case 'line'                 
            % ---------- LINE PLOT ----------
            if (nargin > 1) && ~ischar(varargin{1}),    x = varargin{1};
            else,         x = 1:length(perf);
            end

            cfg = [];
            cfg.label_options   = p.label_options;
            cfg.title_options   = p.title_options;
            cfg.hor             = p.hor;
            cfg.mark_bold       = opt.mask; 
            for ii=1:size(perf,3)  % in case there's multiple results to be plotted
                subplot(1, size(perf,3), ii)
                h.plot(mm) = mv_plot_1D(cfg, x, perf(:,:,ii), perf_std(:,:,ii));
                h.xlabel(mm) = xlabel(p.xlabel, p.label_options{:});
                h.ylabel(mm) = ylabel(p.ylabel, p.label_options{:});
                if p.add_legend, legend(p.legend_labels, p.legend_options{:}), end
                if iscell(p.title)
                    title(p.title{ii}, p.title_options{:});
                else
                    h.title(mm) = title(p.title, p.title_options{:});
                end
            end

        case 'image'                
            % ----------  IMAGE ---------- 
            % apply mask
            if ~isempty(opt.mask)
                perf(~opt.mask) = nan;
            end

            % settings for 2d plot
            cfg= [];
            if (nargin > 1) && ~ischar(varargin{1}), cfg.x = varargin{1};
            else, cfg.x = 1:size(perf,2);
            end
            if (nargin > 2) && ~ischar(varargin{2}), cfg.y = varargin{2};
            else, cfg.y = 1:size(perf,1);
            end
            cfg.climzero    = p.climzero;
            cfg.global_clim = p.global_clim;
            cfg.xlabel      = p.xlabel;
            cfg.ylabel      = p.ylabel;
            cfg.title       = p.title;
            cfg.colorbar_title  = metric;
            cfg.colorbar_location = p.colorbar_location;
            cfg.label_options   = p.label_options;
            cfg.title_options   = p.title_options;
            if p.size_metric_dimension == 1
                h.ax(mm) = mv_plot_2D(cfg, perf);
            else
                h.ax(mm) = mv_plot_2D(cfg, squeeze(permute(perf, [p.plot_dimensions, setdiff(1:3, p.plot_dimensions)])));
            end
    end
    grid on
end



