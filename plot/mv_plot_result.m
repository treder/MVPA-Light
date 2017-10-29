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
%Parameters:
% time              - [N x 1] vector of times representing the x-axis
% dat               - [N x M] data matrix with results. Plots M lines of
%                     length M
% err               - [N x M] data matrix specifying errorbars (optional) 
%                     The external boundedline function is used to plot the
%                     error as a shaded area
%
% cfg          - struct with hyperparameters (use [] to keep all parameters at default):
% xlabel,ylabel     - label for x and y axes (defaults 'Time' and 'Accuracy')
% title             - axis title (default '')
% grid              - options for the grid function (default {'on'})
% lineorder         - order of line types when multiple lines are plotted
%                     (default {'-' '--' ':'})
% hor               - y-value corresponding to horizontal line (default 0.5)
% ver               - x-value corresponding to vertical line (default 0)
% cross             - Give the line options for horizontal and vertical
%                     lines forming a crosshair as cell array (default 
%                     {'--k'}). Set to [] to remove lines
% bounded           - cell array with additional arguments passed to
%                     boundedline.m when a plot with errorbars is created
%                     (default {'alpha'})
%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017

if ~iscell(result), result = {result}; end

nResults = numel(result);
metric = result{1}.metric;
fun = result{1}.function;

fprintf('Plotting the results of %s.\n', result{1}.function);

%% Extract performance metrics
% perf = 

%% Check whether all results have the same classifier
allSameClassifier = numel(unique(cellfun( @(res) res.classifier, result, 'Un', 0)))==1;

%% Extract all performance measures into a matrix
perf = cell2mat(cellfun( @(res) res.perf(:), result, 'Un', 0));
perf_std = cell2mat(cellfun( @(res) res.perf_std(:), result, 'Un', 0));

if strcmp(fun,'mv_classify_timextime')
    perf = cat(3, perf{:});
else
    perf = cat(2, perf{:});
end

%% Get axis or legend labels
lab = arrayfun( @(x) [num2str(x) ' (' result{x}.classifier ')'], 1:nResults,'Un',0);

%% If multiple results are given, calculate mean
if nResults > 1
    hasMean = 1;
    % Find out along which dimension we have to concatenate
    if strcmp(result{1}.function,'mv_classify_timextime')
        if strcmp(result{1}.metrc,'dval')
            cat_dim = 4;
        else
            cat_dim = 3;
        end
    else
        cat_dim = 2;
    end
    
    perf = cat(cat_dim, perf, mean(perf,cat_dim));
    perf_std = cat(cat_dim, perf_std, mean(perf_std,cat_dim));
    nResults = nResults + 1;
    lab = [lab 'MEAN'];
else
    hasMean = 0;
end

%% Struct with handles to graphical objects
h =struct();

%% Plot
clf
switch(fun)
    
    
    %% --------------- MV_CROSSVALIDATE ---------------
    case 'mv_crossvalidate'

        if nResults == 1
            h.bar = bar(perf');
        else
            h.bar = bar(1:nResults, perf');
            set(gca,'XTick',1:nResults, 'XTickLabel',lab)
        end
        
        % Indicate SEM if the bars are not grouped
        if any(strcmp(result{1}.metric,{'auc' 'acc'}))
            hold on
            errorbar(1:nResults,perf',perf_std','.')
        end
        
        % X and Y labels
        h.ylabel = ylabel(result{1}.metric);
        h.fig = gcf;
        h.title = title(result{1}.function,'Interpreter','none');


    %% --------------- MV_CLASSIFY_ACROSS_TIME ---------------
    case 'mv_classify_across_time'
        
        if nargin > 1,  x = varargin{1};
        else,           x = 1:length(result{1}.perf);
        end
        
        cfg = [];
        if any(strcmp(result{1}.metric,{'auc', 'acc'}))
            cfg.hor = 0.5;
        elseif any(strcmp(result{1}.metric,{'dval', 'tval'}))
            cfg.hor = 0;
            
        end
            
        if hasMean
            % If there is a mean, we put it in a separate plot
            title('Single results')
            tmp = mv_plot_1D(cfg,x, perf(:,1:nResults-1), perf_std(:,1:nResults-1) );
            legend(lab(1:nResults-1))
            h.ylabel(1) = ylabel(result{1}.metric);
            h.ax(1) = tmp.ax;
            h.plt = tmp.plt;
            h.fig = gcf;
            h.ylabel(1) = ylabel(result{1}.metric);
            h.title = title(result{1}.function,'Interpreter','none');

            figure
            tmp = mv_plot_1D(cfg,x, perf(:,nResults), perf_std(:,nResults) );
            h.ylabel(2) = ylabel(result{1}.metric);
            h.ax(2) = tmp.ax;
            h.plt = [h.plt; tmp.plt];
            legend({'MEAN'})
            h.fig(2) = gcf;
            h.ylabel(2) = ylabel(result{1}.metric);
            h.title(2) = title([result{1}.function ' (MEAN)'],'Interpreter','none');
            
        else
            tmp = mv_plot_1D(cfg,x, perf, perf_std );
            h.ylabel = ylabel(result{1}.metric);
        end


    %% --------------- MV_CLASSIFY_TIMEXTIME ---------------
    case 'mv_classify_timextime'

    %% --------------- MV_SEARCHLIGHT ---------------
    case 'mv_searchlight'
end

grid on

