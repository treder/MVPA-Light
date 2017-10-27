function h = mv_plot_result(varargin)
%Plots classification results obtained with the functions mv_crossvalidate,
%mv_classify_across_time, mv_classify_timextime, and mv_searchlight.
%
%Usage:
% h = mv_plot_result(result,<...>)
%
%Parameters:
% result            - results struct obtained from one of the
%                     classification functions above. A cell array of
%                     results can be provided (e.g. results for different
%                     subjects). In this case, a single plot is created for
%                     each cell and an additional figure with a grand
%                     average.
%                     
%
%The exact plot depends on which function was used to obtain the results
%
% MV_CROSSVALIDATE:
% h = mv_plot_result(result)
%
% MV_CLASSIFY_ACROSS_TIME:
% h = mv_plot_result(result,x)
%
% MV_CLASSIFY_TIMExTIME:
% h = mv_plot_result(result,x,y)
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
