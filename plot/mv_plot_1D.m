function h = mv_plot_1D(cfg, dat, err)
%Plots 1D results, e.g. classification across time. If a 2D matrix is given 
%several plots are created.
%
%Usage:
% ax = mv_plot_1D(cfg, DAT, ERR)
%
%Parameters:
% DAT               - [N x M] data matrix with results. Plots M lines of
%                     length M
% ERR               - [N x M] data matrix specifying errorbars (optional) 
%                     The external boundedline function is used to plot the
%                     error as a shaded area
%
% cfg          - struct with hyperparameters:
% xlabel,ylabel     - label for x and y axes (default '')
% title             - axis title (default '')
% x                 - x values (e.g. time points)
% grid              - options for the grid function (default {'on'})
% lineorder         - order of line types when multiple lines are plotted
%                     (default {'-' '--' ':'})
% hor               - y-value corresponding to horizontal line (default 0.5)
% ver               - x-value corresponding to vertical line (default 0)
% cross             - Give the line options for horizontal and vertical
%                     lines forming a crosshair as cell array (default 
%                     {'--k'}). Set to [] to remove lines
%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017

[nX,nY] = size(dat);

if nargin<3, err=[]; end

mv_setDefault(cfg,'x',1:nX);
mv_setDefault(cfg,'xlabel','');
mv_setDefault(cfg,'ylabel','');
mv_setDefault(cfg,'title','');
mv_setDefault(cfg,'grid',{'on'});
mv_setDefault(cfg,'lineorder',{'-' '--' ':'});
mv_setDefault(cfg,'hor',0.5);
mv_setDefault(cfg,'ver',0);
mv_setDefault(cfg,'cross',{'--k'});

h = struct();
h.ax = gca;
cla

x= cfg.x;

%% Plot data 
if nargin==3
    % We use boundedline to plot the error as well
    tmp = zeros( size(err,1), 1, size(err,2));
    tmp(:,1,:) = err;
    h.plt = boundedline(cfg.x, dat, tmp);
else
    % Ordinary plot without error
    h.plt = plot(cfg.x, dat);
end
%% Set line styles
for ii=1:nY
    set(h.plt(ii),'LineStyle',cfg.lineorder{ mod(ii-1,numel(cfg.lineorder)) + 1 })
end

%% Mark zero line
if ~isempty(cfg.cross)
    if x(1)<0 && x(end)>0
        hold on
        yl = ylim(gca);
        plot(gca,[1 1] * cfg.ver, yl(:), cfg.cross{:})
        set(gca,'YLim',yl)
    end
    hold on
    xl = xlim(gca);
    plot(gca,xl(:), [1 1] * cfg.hor, cfg.cross{:})
    set(gca,'YLim',yl)
end

%% Add labels and title
xlabel(cfg.xlabel);
ylabel(cfg.ylabel);
title(cfg.title);

%% Add grid
grid(h.ax, cfg.grid{:})

%% Set xlim
xlim([cfg.x(1) cfg.x(end)])


