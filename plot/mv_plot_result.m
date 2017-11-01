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
%Additional arguments can be provided depending on which classification
%function was used to create the results:
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
% Optionally, a second input x can be provided that specifies the values 
% for the x axis (e.g. the times in sec).
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
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017

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

%% Extract all performance measures into a matrix
perf = cellfun( @(res) res.perf, result, 'Un', 0);
perf_std = cellfun( @(res) res.perf_std, result, 'Un', 0);

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

%% Get axis or legend labels
lab = arrayfun( @(x) [num2str(x) ' (' result{x}.classifier ')'], 1:nResults,'Un',0);

%% If multiple results are given, calculate mean
if nResults > 1
    perf_mean = mean(perf,ndims(perf));
    perf_std_mean = mean(perf_std,ndims(perf_std));
    mean_lab = 'MEAN';
end

%% Struct with handles to graphical objects
h =struct();
h.ax = [];
h.title = [];

%% Plot
switch(fun)
    
    
    %% --------------- MV_CROSSVALIDATE ---------------
    case 'mv_crossvalidate'

        figure
        h.ax = gca;
        if nResults == 1
            h.bar = bar(perf');
        else
            h.bar = bar(1:nResults+1, [perf, perf_mean]');
            set(gca,'XTick',1:nResults+1, 'XTickLabel',[lab mean_lab])
        end
        
        % Indicate SEM if the bars are not grouped
        if any(strcmp(metric,{'auc' 'acc'}))
            hold on
            errorbar(1:nResults+1,[perf, perf_mean]', [perf_std, perf_std_mean]','.')
        end
        
        % X and Y labels
        h.ylabel = ylabel(metric);
        h.fig = gcf;
        h.title = title(fun,'Interpreter','none');
        
        % Set Y label
        for ii=1:numel(h.ax)
            h.ylabel(ii) = ylabel(h.ax(ii),metric);
        end

    %% --------------- MV_CLASSIFY_ACROSS_TIME ---------------
    case 'mv_classify_across_time'
        
        if nargin > 1,  x = varargin{1};
        else,           x = 1:length(result{1}.perf);
        end
        
        figure
        cfg = [];
        if any(strcmp(metric,{'auc', 'acc'}))
            cfg.hor = 0.5;
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
                legend(lab(ii))
                h.ax = [h.ax; tmp.ax];
                h.plt = [h.plt; tmp.plt];
                h.fig = gcf;
                h.title = [h.title; title(fun,'Interpreter','none')];
            end
        else
            tmp = mv_plot_1D(cfg,x, perf, perf_std);
            legend(lab)
            h.ax = tmp.ax;
            h.plt = tmp.plt;
            h.fig = gcf;
            h.title = title(fun,'Interpreter','none');
        end
        
        % Plot mean
        if nResults > 1
            figure
            tmp = mv_plot_1D(cfg,x, perf_mean, perf_std_mean );
            set(tmp.plt, 'LineWidth',2);
            h.ax = [h.ax; tmp.ax];
            h.plt = [h.plt; tmp.plt];
            legend({'MEAN'})
            h.fig(2) = gcf;
            h.title = [h.title; title([fun ' (MEAN)'],'Interpreter','none')];
        end

        % Set Y label
        for ii=1:numel(h.ax)
            h.ylabel(ii) = ylabel(h.ax(ii),metric);
        end

    %% --------------- MV_CLASSIFY_TIMEXTIME ---------------
    case 'mv_classify_timextime'

        if nargin > 1,  x = varargin{1};
        else,           x = 1:size(result{1}.perf,1);
        end
        if nargin > 2,  y = varargin{2};
        else,           y = 1:size(result{1}.perf,2);
        end
        
        % settings for 2d plot
        cfg= [];
        cfg.x   = x;
        cfg.y   = y;
        if any(strcmp(metric,{'auc', 'acc'}))
            cfg.climzero = 0.5;
        elseif any(strcmp(metric,{'dval', 'tval'}))
            cfg.climzero = 0;
        end
        
        if strcmp(metric,'dval')
            % dval: create figure each class
            hs = cell(1,2);
            for cl=1:2
                figure
                cfg.title = strcat(fun, ' - ' ,lab, ' (class ', num2str(cl),')');
                hs{cl} = mv_plot_2D(cfg, squeeze(perf(:,cl,:,:)) );
            end
            h = [hs{:}];
            
            % Plot mean
            if nResults > 1
                figure
                cfg.title = strcat(fun, '-' ,mean_lab);
                h(numel(h)+1) = mv_plot_2D(cfg, cat(3,squeeze(perf_mean(:,1,:)), squeeze(perf_mean(:,2,:))) );
            end
        else
            cfg.title = strcat(fun, '-' ,lab);
            h = mv_plot_2D(cfg, perf);
            
            % Plot mean
            if nResults > 1
                figure
                cfg.title = strcat(fun, '-' ,mean_lab);
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
                    cfg.title = strcat(fun, ' - ' ,lab, ' (class ', num2str(cl),')');
                    hs{cl} = mv_plot_topography(cfg, squeeze(perf(:,cl,:)), chans.pos);
                    axis on
                    axis off
                end
                h = [hs{:}];
            else
                % no dval: all plots in one figure
                cfg.title = strcat(fun, ' - ' ,lab);
                h = mv_plot_topography(cfg, perf, chans.pos);
            end
            
            % Plot mean
            if nResults > 1
                figure
                cfg.title = strcat(fun, '-' ,mean_lab);
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
                    hs{cl}.title= title(sprintf('%s - class %d',fun,cl),'Interpreter','none');
                    hs{cl}.xlabel = xlabel('features');
                    hs{cl}.ylabel = ylabel(metric);
                    set(gca,'XTick',1:nResults,'XTickLabel',lab)
                end
                h = [hs{:}];
            else
                figure
                h.bar = bar(perf');
                h.xlabel = xlabel('features');
                h.ylabel = ylabel(metric);
                set(gca,'XTick',1:nResults,'XTickLabel',lab)
                h.title= title(fun,'Interpreter','none');
            end
            grid on
            
            % Plot mean
            if nResults > 1
                figure
                h(2).bar = bar(perf_mean');
                h(2).xlabel = xlabel('features');
                h(2).ylabel = ylabel(metric);
                h(2).title= title(strcat(fun, '-' ,mean_lab),'Interpreter','none');
            end
        end
        
end

grid on
