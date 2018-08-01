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
% Usage: h = mv_plot_result(result)
%
% Plots the classification result as a barplot. Plots multiple bars and the
% mean, if multiple result are provided. 
%
% MV_CLASSIFY_ACROSS_TIME:
% Usage: h = mv_plot_result(result,x)
%
% Plots the classification result as a line plot. Plots multiple lines and 
% a mean, if multiple result are provided. 
%
% MV_CLASSIFY_TIMExTIME:
% h = mv_plot_result(result,x,y)
%
% Plots the classification result as an image. Plots multiple images and a
% mean, if multiple result are provided.
% Optionally, second and third inputs x and y can be provided that 
% specify the values for the x and y axes (e.g. the times in sec).
%
% MV_SEARCHLIGHT:
% h = mv_plot_result(result,chanlocs)
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
% plot_mean      - if 1 and multiple results are provided, also plots the 
%                  mean across the results (default 1)
% new_figure     - if 1, results are plotted in a new figure. If 0, results
%                  are plotted in the current axes instead (default 1)
%
% RETURNS:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017-2018

if ~iscell(result), result = {result}; end

nResults = numel(result);
metric = result{1}.metric;
fun = result{1}.function;

if numel(unique(cellfun( @(res) res.function, result,'Un',0))) > 1
    error('All results must come from the same function')
end
if numel(unique(cellfun( @(res) res.metric, result,'Un',0))) > 1
    error('All results must use the same metric')
end

fprintf('Plotting the results of %s.\n', fun);

%% Parse any key-value pairs
opt = mv_parse_key_value_pairs(varargin{:});

if ~isfield(opt,'plot_mean'), opt.plot_mean = 1; end
if ~isfield(opt,'new_figure'), opt.new_figure = 0; end

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

if strcmp(metric,'dval')
    cat_dim = cat_dim + 1;
end

perf = cat(cat_dim, perf{:});
perf_std = cat(cat_dim, perf_std{:});

%% Create axis or legend labels (unless they have already been specified)
if ~isfield(opt,'label')
    opt.label = arrayfun( @(x) [num2str(x) ' (' result{x}.classifier ')'], 1:nResults,'Un',0);
end

%% If multiple results are given, calculate mean
opt.plot_mean = (opt.plot_mean && nResults > 1);
if opt.plot_mean
    perf_mean = mean(perf,ndims(perf));
    perf_std_mean = mean(perf_std,ndims(perf_std));
    mean_lab = 'MEAN';
end

%% Struct with handles to graphical objects
h =struct();
h.ax = [];
h.title = [];

%% Prepare title
titleopt = {'Interpreter','none'};
if ~isfield(opt,'title')
    opt.title = fun;
end

%% Plot
switch(fun)
    
    
    %% --------------- MV_CROSSVALIDATE ---------------
    case 'mv_crossvalidate'

        if opt.new_figure, figure; end
        h.ax = gca;
        if nResults == 1
            h.bar = bar(perf');
        else
            h.bar = bar(1:nResults+1, [perf, perf_mean]');
            set(gca,'XTick',1:nResults+1, 'XTickLabel',[opt.label mean_lab])
        end
        
        % Indicate SEM if the bars are not grouped
        if any(strcmp(metric,{'auc' 'acc'}))
            hold on
            errorbar(1:nResults+1,[perf, perf_mean]', [perf_std, perf_std_mean]','.')
        end
        
        % X and Y labels
        h.ylabel = ylabel(metric);
        h.fig = gcf;
        h.title = title(opt.title,titleopt{:});
        
        % Set Y label
        for ii=1:numel(h.ax)
            h.ylabel(ii) = ylabel(h.ax(ii),metric);
        end

    %% --------------- MV_CLASSIFY_ACROSS_TIME ---------------
    case 'mv_classify_across_time'
        
        if nargin > 1,  x = varargin{1};
        else,           x = 1:length(result{1}.perf);
        end
        
        if opt.new_figure, figure; end
        cfg = [];
        if any(strcmp(metric,{'auc', 'acc'}))
            cfg.hor = 1 / result{1}.nclasses;
        elseif any(strcmp(metric,{'dval', 'tval'}))
            cfg.hor = 0;
        end
            
        if strcmp(metric,'dval')
            % dval: create separate subplot for each result
            N = size(perf,3);
            nc = ceil(sqrt(N));
            nr = ceil(N/nc);
            h.plt = [];
            for ii=1:N
                subplot(nr,nc,ii)
                tmp = mv_plot_2D(cfg,x, squeeze(perf(:,:,ii)), squeeze(perf_std(:,:,ii)) );
                legend(opt.label(ii))
                h.ax = [h.ax; tmp.ax];
                h.plt = [h.plt; tmp.plt];
                h.fig = gcf;
                h.title = [h.title; title(opt.title,titleopt{:})];
            end
        else
            tmp = mv_plot_1D(cfg,x, perf, perf_std);
            legend(opt.label)
            h.ax = tmp.ax;
            h.plt = tmp.plt;
            h.fig = gcf;
            h.title = title(opt.title,titleopt{:});
        end
        
        % Plot mean
        if opt.plot_mean
            figure
            tmp = mv_plot_1D(cfg,x, perf_mean, perf_std_mean);
            set(tmp.plt, 'LineWidth',2);
            h.ax = [h.ax; tmp.ax];
            h.plt = [h.plt; tmp.plt];
            legend({'MEAN'})
            h.fig(2) = gcf;
            h.title = [h.title; title([opt.title ' (MEAN)'],titleopt{:})];
        end

        % Set Y label
        for ii=1:numel(h.ax)
            h.ylabel(ii) = ylabel(h.ax(ii),metric);
        end

    %% --------------- MV_CLASSIFY_TIMEXTIME ---------------
    case 'mv_classify_timextime'

%         if nargin > 1,  x = varargin{1};
%         else,           x = 1:size(result{1}.perf,1);
%         end
%         if nargin > 2,  y = varargin{2};
%         else,           y = 1:size(result{1}.perf,2);
%         end
        
        % settings for 2d plot
        cfg= [];
        if isfield(opt,'x'), cfg.x = opt.x; end
        if isfield(opt,'y'), cfg.y = opt.y; end
        if any(strcmp(metric,{'auc', 'acc'}))
            cfg.climzero = 1 / result{1}.nclasses;
        elseif any(strcmp(metric,{'dval', 'tval'}))
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
            
            % Plot mean
            if opt.plot_mean
                figure
                cfg.title = strcat(opt.title, '-' ,mean_lab);
                h(numel(h)+1) = mv_plot_2D(cfg, cat(3,squeeze(perf_mean(:,1,:)), squeeze(perf_mean(:,2,:))) );
            end
        else
            cfg.title = strcat(opt.title, '-' ,opt.label);
            h = mv_plot_2D(cfg, perf);
            
            % Plot mean
            if opt.plot_mean
                figure
                cfg.title = strcat(opt.title, '-' ,mean_lab);
                h(numel(h)+1) = mv_plot_2D(cfg, perf_mean );
            end
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
            
            if any(strcmp(metric,{'auc', 'acc'}))
                cfg.climzero = 0.5;
            elseif any(strcmp(metric,{'dval', 'tval'}))
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
            
            % Plot mean
            if opt.plot_mean
                figure
                cfg.title = strcat(opt.title, '-' ,mean_lab);
                h(numel(h)+1) = mv_plot_topography(cfg, perf_mean, chans.pos);
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
                    set(gca,'XTick',1:nResults,'XTickLabel',opt.label)
                end
                h = [hs{:}];
            else
                if opt.new_figure, figure; end
                h.bar = bar(perf');
                h.xlabel = xlabel('features');
                h.ylabel = ylabel(metric);
                set(gca,'XTick',1:nResults,'XTickLabel',opt.label)
                h.title= title(opt.title,titleopt{:});
            end
            grid on
            
            % Plot mean
            if opt.plot_mean
                figure
                h(2).bar = bar(perf_mean');
                h(2).xlabel = xlabel('features');
                h(2).ylabel = ylabel(metric);
                h(2).title= title(strcat(opt.title, '-' ,mean_lab),titleopt{:});
            end
        end
        
end

grid on

end