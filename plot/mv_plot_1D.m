function h = mv_plot_1D(varargin)
%Plots 1D results, e.g. classification across time. If a 2D matrix is given 
%several plots are created.
%
%Usage: Two possible usages, either giving additional parameters in a cfg
%       struct (with cfg.key1 = value1) or directly giving the key-value 
%       pairs as parameters:
% ax = mv_plot_1D(cfg, xval, dat, err)
% ax = mv_plot_1D(xval, dat, err, key1, value1, key2, value2, ...)
%
%Parameters:
% xval              - [N x 1] vector of values representing the x-axis
% dat               - [N x M] data matrix with results. Plots M lines of
%                     length M
% err               - [N x M] data matrix specifying errorbars (optional) 
%                     The external boundedline function is used to plot the
%                     error as a shaded area. Can be set to [] if no 
%                     errorbars is given but key-value pairs are to be
%                     provided
%
% cfg          - struct with additional parameters (use [] to keep all parameters at default)
%                Alternatively, the parameters can be presented as
%                key-value pairs.
%
% The additional parameters are given here:
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
% mark_bold         - when a binary mask is provided (eg with statistical
%                     significance) the corresponding lines will be
%                     plotted bold
%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder

has_errorbar = 0;

if isstruct(varargin{1}) || isempty(varargin{1})
    % Additional parameters are specified in struct cfg
    cfg = varargin{1};
    xval = varargin{2};
    dat = varargin{3};
    if nargin < 4, 	err = []; 
    else,           err = varargin{4}; end
else
    % Additional parameters are given as key-value pairs
    xval = varargin{1};
    dat = varargin{2};
    if nargin < 3, 	err = []; 
    else,           err = varargin{3}; end
    if nargin > 3
        cfg = mv_parse_key_value_pairs(varargin{4:end});
    else
        cfg = [];
    end
end
if ~isempty(err), has_errorbar = 1; end

[nX,nY] = size(dat);

mv_set_default(cfg,'xlabel','Time');
mv_set_default(cfg,'ylabel','Accuracy');
mv_set_default(cfg,'title','');
mv_set_default(cfg,'grid',{'on'});
mv_set_default(cfg,'lineorder',{'-' '-' '--' '--' ':'});
mv_set_default(cfg,'hor',0.5);
mv_set_default(cfg,'ver',0);
mv_set_default(cfg,'cross',{'--k'});
mv_set_default(cfg,'bounded',{'alpha'});
mv_set_default(cfg,'mark_bold',[]);

mv_set_default(cfg,'label_options ', {'Fontsize', 14});
mv_set_default(cfg,'title_options', {'Fontsize', 16, 'Fontweight', 'bold'});

h = struct();
h.ax = gca;
cla

%% Plot data 
if has_errorbar
    % We use boundedline to plot the error as well
    tmp = zeros( size(err,1), 1, size(err,2));
    tmp(:,1,:) = err;
    [h.plt, h.patch] = boundedline(xval, dat, tmp, cfg.bounded{:});
else
    % Ordinary plot without errorbars
    h.plt = plot(xval, dat);
end

%% Set line styles
for ii=1:nY
    set(h.plt(ii),'LineStyle',cfg.lineorder{ mod(ii-1,numel(cfg.lineorder)) + 1 })
end

%% mark parts of the data using a bold line
if ~isempty(cfg.mark_bold)
    dat(~cfg.mark_bold) = nan;
    tmp_h = boundedline(xval, dat, tmp, cfg.bounded{:});
    set(tmp_h,'LineWidth', 4*get(tmp_h,'LineWidth'));
end

%% Mark zero line
if ~isempty(cfg.cross)
    if xval(1)<0 && xval(end)>0 && ~isempty(cfg.ver)
        % vertical zero line
        hold on
        yl = ylim(gca);
        h.vertical_line = plot(gca,[1 1] * cfg.ver, yl(:), cfg.cross{:});
        set(gca,'YLim',yl)
    end
    hold on
    
    % horizontal zero line
    if ~isempty(cfg.hor)
        xl = xlim(gca);
        h.horizontal_line = plot(gca,xl(:), [1 1] * cfg.hor, cfg.cross{:});
    end
%     set(gca,'YLim',yl)
end

%% Add labels and title
xlabel(cfg.xlabel, cfg.label_options{:});
ylabel(cfg.ylabel, cfg.label_options{:});
title(cfg.title, cfg.title_options{:});

%% Add grid
grid(h.ax, cfg.grid{:})

%% Set xlim
xlim([xval(1) xval(end)])


