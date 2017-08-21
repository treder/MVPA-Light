function h = mv_plot_1D(cfg, time, dat, err)
%Plots 1D results, e.g. classification across time. If a 2D matrix is given 
%several plots are created.
%
%Usage:
% ax = mv_plot_1D(cfg, time, dat, err)
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

[nX,nY] = size(dat);

if nargin<3, err=[]; end

mv_setDefault(cfg,'xlabel','Time');
mv_setDefault(cfg,'ylabel','Accuracy');
mv_setDefault(cfg,'title','');
mv_setDefault(cfg,'grid',{'on'});
mv_setDefault(cfg,'lineorder',{'-' '-' '--' '--' ':'});
mv_setDefault(cfg,'hor',0.5);
mv_setDefault(cfg,'ver',0);
mv_setDefault(cfg,'cross',{'--k'});
mv_setDefault(cfg,'bounded',{'alpha'});

h = struct();
h.ax = gca;
cla

%% Plot data 
if nargin==4
    % We use boundedline to plot the error as well
    tmp = zeros( size(err,1), 1, size(err,2));
    tmp(:,1,:) = err;
    h.plt = boundedline(time, dat, tmp, cfg.bounded{:});
else
    % Ordinary plot without error
    h.plt = plot(time, dat);
end
%% Set line styles
for ii=1:nY
    set(h.plt(ii),'LineStyle',cfg.lineorder{ mod(ii-1,numel(cfg.lineorder)) + 1 })
end

%% Mark zero line
if ~isempty(cfg.cross)
    if time(1)<0 && time(end)>0
        hold on
        yl = ylim(gca);
        plot(gca,[1 1] * cfg.ver, yl(:), cfg.cross{:})
        set(gca,'YLim',yl)
    end
    hold on
    xl = xlim(gca);
    plot(gca,xl(:), [1 1] * cfg.hor, cfg.cross{:})
%     set(gca,'YLim',yl)
end

%% Add labels and title
xlabel(cfg.xlabel);
ylabel(cfg.ylabel);
title(cfg.title);

%% Add grid
grid(h.ax, cfg.grid{:})

%% Set xlim
xlim([time(1) time(end)])


