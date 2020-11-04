function h = mv_plot_result(result, varargin)
%Provides a simple visual representation of the results obtained with the
%functions mv_crossvalidate, mv_classify_across_time, mv_classify_timextime, 
%and mv_searchlight. 
%
%The type of plot depends on which of these functions was used. 
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
% Additional arguments can be provided as key-value parameters, e.g.
% mv_plot_result(result,'title','This is my title'). See ADDITIONAL 
% KEY-VALUE ARGUMENTS below.
% 
% Furthermore, additional arguments can be provided depending on which 
% classification function was used to create the results, as described
% next:
%
% MV_CROSSVALIDATE:
% Usage: h = mv_plot_result(result,...)
%
% Plots the classification result as a barplot. Plots multiple bars and the
% mean, if multiple result are provided. 
%
% MV_CLASSIFY_ACROSS_TIME:
% Usage: h = mv_plot_result(result,x,...)
%
% Plots the classification result as a line plot. Plots multiple lines and 
% a mean, if multiple result are provided.  Optional in x is the values for
% the x axis (eg time in sec).
%
% MV_CLASSIFY_TIMExTIME:
% h = mv_plot_result(result,x,y,...)
%
% Plots the classification result as an image. Plots multiple images and a
% mean, if multiple result are provided.
% Optionally, second and third inputs x and y can be provided that 
% specify the values for the x and y axes (eg time in sec).
%
% MV_SEARCHLIGHT:
% h = mv_plot_result(result,chanlocs,...)
%
% Plots classification performance for each feature. 
% If the features correspond to EEG/MEG channels and channel locations are
% provided (chanlocs must be a struct with a field pos specifying their 2d
% positions) the performance is plotted as a topography.
% In any other case, the features are plotted as bars in a bar graph.
%
% ADDITIONAL KEY-VALUE ARGUMENTS:
% title          - string that serves as axis title
% label          - if result argument is a cell array, a cell array of strings can
%                  be provided to label the different results (serves as
%                  legend labels for mv_classify_across_time plots and as
%                  xlabels for mv_crossvalidate plots)
% new_figure     - if 1, results are plotted in a new figure. If 0, results
%                  are plotted in the current axes instead (default 1)
%
% RETURNS:
% h        - struct with handles to the graphical elements 

% (c) matthias treder

if ~iscell(result), result = {result}; end

if iscell(result{1}.metric) && numel(result{1}.metric) > 1
    % if multiple metrics are provided, this function is called for each
    % metric separately
    for mm=1:numel(result{1}.metric)
        res = cell(numel(result), 1);
        for ii=1:numel(result)
            res{ii} = result{ii};
            res{ii}.metric = res{ii}.metric{mm};
            res{ii}.perf = res{ii}.perf{mm};
            res{ii}.perf_std = res{ii}.perf_std{mm};
        end
        mv_plot_result(res, varargin{:});
    end
    h=[];
    return
end

nresults = numel(result);
nclasses = result{1}.nclasses;
metric = result{1}.metric;

fun = result{1}.function;

if numel(unique(cellfun( @(res) res.function, result,'Un',0))) > 1
    error('All results must come from the same function')
end
if numel(unique(cellfun( @(res) res.metric, result,'Un',0))) > 1
    error('All results must use the same metric')
end

fprintf('Plotting the results of %s.\n', fun);

%% Check whether the combination of metric and function is supported for plotting
incompatible_metric_function = {...
    'confusion'          {'mv_classify_across_time','mv_classify_timextime','mv_searchlight'};
    };

idx = find(ismember(incompatible_metric_function(:,1), result{1}.metric));

if any(idx) && any(ismember(incompatible_metric_function{idx,2}, result{1}.function))
    error('mv_plot_result does not currently support the metric ''%s'' for results from %s', incompatible_metric_function{idx,1}, result{1}.function)
end

%% Parse any key-value pairs
opt = mv_parse_key_value_pairs(varargin{:});

if ~isfield(opt,'new_figure'), opt.new_figure = 1; end

%% Extract all performance measures into a matrix
perf = cellfun( @(res) res.perf, result, 'Un', 0);
if ~isempty(result{1}.perf_std)
    perf_std = cellfun( @(res) res.perf_std, result, 'Un', 0);
else
    perf_std = cellfun( @(res) res.perf * 0, result, 'Un', 0);
end

if strcmp(fun,'mv_classify_timextime')
    cat_dim = 3;
else
    cat_dim = 2;
end

% check whether metric is multivariate
is_multivariate = strcmp(metric,'dval') || ...
        (nclasses>2 && any(strcmp(metric,{'precision','recall','f1'})));
    
if is_multivariate
    cat_dim = cat_dim + 1;
end

perf = cat(cat_dim, perf{:});
perf_std = cat(cat_dim, perf_std{:});

%% Create axis or legend labels (unless they have already been specified)
if ~isfield(opt,'label')
    if nresults==1
        opt.label = {result{1}.classifier};
    else
        opt.label = arrayfun( @(x) [num2str(x) ' (' result{x}.classifier ')'], 1:nresults,'Un',0);
    end
end

%% Struct with handles to graphical objects
h =struct();
h.ax = [];
h.title = [];

%% Prepare title
titleopt = {'Interpreter','none'};
if ~isfield(opt,'title')
    opt.title = metric;
end

% Plotting options 
leg_opt = {'Interpreter','none'};

%% Plot
switch(fun)
    %% --------------- MV_CROSSVALIDATE ---------------
    case 'mv_crossvalidate'

        if opt.new_figure, figure; end
        h.ax = gca;
        if strcmp(result{1}.metric, 'confusion')  % plot confusion matrix
            opt_txt = {'Fontsize',15,'HorizontalAlignment','center'};
            for ii=1:nresults
                subplot(1,nresults,ii)
                imagesc(result{ii}.perf)
                colorbar
                h.xlabel(ii) = xlabel('Predicted class');
                h.ylabel(ii) = ylabel('True class');
                set(gca,'Xtick',1:nclasses,'Ytick',1:nclasses)
                for rr=1:nclasses
                    for cc=1:nclasses
                        text(cc,rr, sprintf('%0.2f',result{ii}.perf(rr,cc)), opt_txt{:})
                    end
                end
                h.title(ii) = title(sprintf('confusion matrix\n%s',result{ii}.classifier),titleopt{:});
            end
        else
            h.bar = bar(1:nresults, perf');
            set(gca,'XTick',1:nresults, 'XTickLabel', strrep(opt.label,'_','\_'))
            % Indicate SEM if the bars are not grouped
            if any(strcmp(metric,{'auc' 'acc' 'accuracy' 'precision' 'recall' 'f1'}))
                hold on
                errorbar(1:nresults, perf', perf_std','.')
            end
            
            % X and Y labels
            h.ylabel = ylabel(metric);
            h.fig = gcf;
            h.title = title(opt.title,titleopt{:});
            
            % Set Y label
            for ii=1:numel(h.ax)
                h.ylabel(ii) = ylabel(h.ax(ii),metric);
            end
        end

    %% --------------- MV_CLASSIFY_ACROSS_TIME ---------------
    case 'mv_classify_across_time'
        
        if (nargin > 1) && ~ischar(varargin{1})
            x = varargin{1};
        else
            x = 1:length(result{1}.perf);
        end
        
        if opt.new_figure, figure; end
        if any(strcmp(metric,{'auc', 'acc','accuracy','precision','recall','f1'}))
            hor = 1 / nclasses;
        elseif any(strcmp(metric,{'dval', 'tval','kappa'}))
            hor = 0;
        end
            
        opt_1D = {'ylabel',metric, 'hor', hor};
        if is_multivariate
            % dval: create separate subplot for each result
            N = size(perf,3);
            nc = ceil(sqrt(N));
            nr = ceil(N/nc);
            h.plt = [];
            classes = arrayfun( @(x) sprintf('Class %d', x), 1:nclasses, 'Un', 0);
            for ii=1:N
                subplot(nr,nc,ii)
                tmp = mv_plot_1D(x, squeeze(perf(:,:,ii)), squeeze(perf_std(:,:,ii)), opt_1D{:});
                legend(tmp.plt, classes,leg_opt{:})
                h.ax = [h.ax; tmp.ax];
                h.plt = [h.plt; tmp.plt];
                h.fig = gcf;
                if N>1
                    h.title = [h.title; title(sprintf('%s [%d]',opt.title,ii),titleopt{:})];
                else
                    h.title = [h.title; title(sprintf('%s',opt.title),titleopt{:})];
                end
            end
        else
            tmp = mv_plot_1D(x, perf, perf_std,opt_1D{:});
            legend(opt.label,leg_opt{:})
            h.ax = tmp.ax;
            h.plt = tmp.plt;
            h.fig = gcf;
            h.title = title(opt.title,titleopt{:});
        end
        
        % Set Y label
        for ii=1:numel(h.ax)
            h.ylabel(ii) = ylabel(h.ax(ii),metric);
        end

    %% --------------- MV_CLASSIFY_TIMEXTIME ---------------
    case 'mv_classify_timextime'
        
        % settings for 2d plot
        cfg= [];
        if (nargin > 1) && ~ischar(varargin{1}), cfg.x = varargin{1};
        else, cfg.x = 1:size(result{1}.perf,1);
        end
        if (nargin > 2) && ~ischar(varargin{2}), cfg.y = varargin{2};
        else, cfg.y = 1:size(result{1}.perf,2);
        end
        if any(strcmp(metric,{'auc', 'acc','accuracy','f1'}))
            cfg.climzero = 1 / nclasses;
        elseif any(strcmp(metric,{'dval', 'tval','precision','recall','kappa'}))
            cfg.climzero = 0;
        end
        
        if strcmp(metric,'dval')
            % dval: create figure each class
            hs = cell(1,2);
            for cl=1:2
                figure
                cfg.title = strcat(opt.title, ' - ' ,opt.label, ' (class ', num2str(cl),')');
                hs{cl} = mv_plot_2D(cfg, squeeze(perf(:,cl,:,:)) );
            end
            h = [hs{:}];
            
        else
            cfg.title = strcat(opt.title, '-' ,opt.label);
            h = mv_plot_2D(cfg, perf);
        end
        
        % set metric as title for colorbar
        for ii=1:numel(h)
            set(get(h(ii).colorbar,'title'),'String',metric)
        end


    %% --------------- MV_SEARCHLIGHT ---------------
    case 'mv_searchlight'
       
        if nargin>1
            % If a struct with channel information is given, we use it to plot
            % a topography
            chans = varargin{1};
            cfg = [];
            cfg.cbtitle = metric;
            cfg.clim = 'sym';
            
            if any(strcmp(metric,{'auc', 'acc','accuracy'}))
                cfg.climzero = 0.5;
            elseif any(strcmp(metric,{'dval', 'tval','precision','recall','f1'}))
                cfg.climzero = 0;
            end

            if isfield(chans,'outline'), cfg.outline = chans.outline; end
            
            if strcmp(metric,'dval')
                % dval: create figure each class
                hs = cell(1,2);
                for cl=1:2
                    figure
                    cfg.title = strcat(opt.title, ' - ' ,opt.label, ' (class ', num2str(cl),')');
                    hs{cl} = mv_plot_topography(cfg, squeeze(perf(:,cl,:)), chans.pos);
                    axis on
                    axis off
                end
                h = [hs{:}];
            else
                % no dval: all plots in one figure
                cfg.title = strcat(opt.title, ' - ' ,opt.label);
                h = mv_plot_topography(cfg, perf, chans.pos);
            end
            
        else
            % If no chans are provided: plot classification performance 
            % for each feature as a grouped bar graph
            
            if strcmp(metric,'dval')
                % dval: create figure each class
                hs = cell(1,2);
                for cl=1:2
                    figure
                    hs{cl}.bar = bar(squeeze(perf(:,cl,:))');
                    hs{cl}.title= title(sprintf('%s - class %d',opt.title,cl),titleopt{:});
                    hs{cl}.xlabel = xlabel('features');
                    hs{cl}.ylabel = ylabel(metric);
                    set(gca,'XTick',1:nresults,'XTickLabel',opt.label)
                end
                h = [hs{:}];
            else
                if opt.new_figure, figure; end
                h.bar = bar(perf');
                h.xlabel = xlabel('features');
                h.ylabel = ylabel(metric);
                set(gca,'XTick',1:nresults,'XTickLabel',opt.label)
                h.title= title(opt.title,titleopt{:});
            end
            grid on
            
        end
        
end

grid on

end