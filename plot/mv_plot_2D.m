function h = mv_plot_2D(varargin)
%Plots 2D results, e.g. a time x time generalisation. Plots the results as
%a color matrix. If a 3D matrix is given several subplots are created.
%
%%Usage: Two possible usages, either giving additional parameters in a cfg
%       struct (with cfg.key1 = value1) or directly giving the key-value 
%       pairs as parameters:
% ax = mv_plot_2D(cfg, dat)
% ax = mv_plot_2D(dat, key1, value1, key2, value2, ...)
%
%Parameters:
% DAT               - [N x M] data matrix with results or [N x M x P] 3D
%                     matrix with P different images. For multiple images,
%                     all images need to have the same x and y axis 
%
% cfg          - struct with additional parameters (use [] to keep all parameters at default)
%                Alternatively, the parameters can be presented as
%                key-value pairs.
% 
% The additional parameters are given here:
% xlabel,ylabel     - label for x and y axes (default 'Training time' and
%                     'Testing time')
% title             - axis title (default '')
% x, y              - x and y values (e.g. time points, frequencies). the 
%                     number of elements in x and y should match the size of DAT
% xlim,ylim         - [min val, max val] vector that specifies the x axis
%                     and y axis limits
% grid              - options for the grid function (default {'on'})
% clim              - set to fix the color limits
%                     [cmin cmax] sets the color limits manually
%                     'minmax' uses the minimum and maximum values in the
%                     plot
%                     'sym' makes the color values symmetric about a
%                     reference point (e.g. [-0.7, 0.7] would be symmetric
%                     about 0. climzero should be set if 'sym' is selected
%                     (default 'sym')
% climzero          - if clim = 'sym' a "zero-point" needs to be defined
%                     (for instance 0.5 for classification accuracies)
%                     (default 0.5)
% global_clim       - if 1 equalises the clim across all plots (if P>1)
%                     (default 1)
% zero              - marks the zero point with a horizontal and vertical
%                     line. Give the line options as cell array (default 
%                     {'--k'}). Set to [] to remove lines
% nrow              - if a 3D matrix is given: number of rows for subplot
%                     (default 1)
% ncol              - if a 3D matrix is given: number of columns for subplot
%                     (default P = size of 3rd dimension of R)
% xscale,yscale     - scaling of the x and y axes, 'linear' (default) or
%                     'log'. Note that log scaling might not render the
%                     image. This happens if your x and y contains negative
%                     values or values close to 0, since the values are
%                     taken as the centers of the image pixels (the border
%                     of the pixel could be <= 0, makes it impossible to
%                     render on a log scale)
% title_options, label_options  - key/value pairs with options
%
% Specific options for the COLORBAR:
% colorbar          - if 1 a colorbar is added to each plot. If there is
%                     multiple plots and global_clim=1, a single colorbar is
%                     plotted if the colorbar location is outside (default 1)
% colorbar_location - colorbar location, see help colorbar (default
%                     'EastOutside')
% colorbar_title    - [string] title for colorbar
%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017-2018

if isstruct(varargin{1}) || isempty(varargin{1})
    % Additional parameters are specified in struct cfg
    cfg = varargin{1};
    dat = varargin{2};
else
    % Additional parameters are given as key-value pairs
    dat = varargin{1};
    if nargin > 1
        cfg = mv_parse_key_value_pairs(varargin{2:end});
    else
        cfg = struct();
    end
end

[nX,nY,P] = size(dat);

mv_set_default(cfg,'x',1:nY);
mv_set_default(cfg,'y',1:nX);
mv_set_default(cfg,'xlim',[min(cfg.x), max(cfg.x)]);
mv_set_default(cfg,'ylim',[min(cfg.y), max(cfg.y)]);
mv_set_default(cfg,'xlabel','Testing time');
mv_set_default(cfg,'ylabel','Training time');
mv_set_default(cfg,'title','');
mv_set_default(cfg,'clim','sym');
mv_set_default(cfg,'climzero',0.5);
mv_set_default(cfg,'global_clim',1);
mv_set_default(cfg,'grid',{'on'});
mv_set_default(cfg,'zero',{'--k'});
mv_set_default(cfg,'ncol',ceil(sqrt(P)));
mv_set_default(cfg,'nrow',ceil(P/cfg.ncol));
mv_set_default(cfg,'colorbar',1);
mv_set_default(cfg,'colorbar_location','South');
mv_set_default(cfg,'colorbar_title','');
mv_set_default(cfg,'ydir','normal');
mv_set_default(cfg,'yscale','linear');
mv_set_default(cfg,'xscale','linear');

mv_set_default(cfg,'label_options ', {'Fontsize', 14});
mv_set_default(cfg,'title_options', {'Fontsize', 16, 'Fontweight', 'bold'});
mv_set_default(cfg,'colorbar_title_options', {'Fontsize', 12 ,'Fontweight', 'normal'});


if ~iscell(cfg.grid), cfg.grid={cfg.grid}; end
if ~iscell(cfg.xlabel)
    cfg.xlabel = repmat({cfg.xlabel},[1 P]);
end
if ~iscell(cfg.ylabel)
    cfg.ylabel = repmat({cfg.ylabel},[1 P]);
end
if ~iscell(cfg.title)
    cfg.title = repmat({cfg.title},[1 P]);
end

h = struct();
h.ax = gca;

%% Select samples according to xlim and ylim
xsel = find(cfg.x >= cfg.xlim(1)  & cfg.x <= cfg.xlim(2));
ysel = find(cfg.y >= cfg.ylim(1)  & cfg.y <= cfg.ylim(2));

x= cfg.x(xsel);
y= cfg.y(ysel);

%% Plot data matrix
axnum=1;
for rr=1:cfg.nrow
    for cc=1:cfg.ncol
        if axnum<=P
            if P > 1
                h.ax(axnum) = subplot(cfg.nrow,cfg.ncol,axnum);
            else
                h.ax = gca;
            end
            set(gcf,'CurrentAxes',h.ax(axnum));
            
            % Plot the classification performance image here. The y-axis
            % represents training time and the x-axis represents testing
            % time
            imagesc(x, y, squeeze(dat(ysel,xsel,axnum)));
            axnum=axnum+1;
        end
    end
end

%% Set color limits
cl = get(h.ax,'CLim');
if iscell(cl), cl = [cl{:}]; end

if isnumeric(cfg.clim) && numel(cfg.clim)==2
    set(h.ax,'CLim',cfg.clim);
elseif strcmp(cfg.clim,'maxmin') 
    % this is the default - we need only need to do smth when global clim
    % is requestedxt{ii}
    if cfg.global_clim
        set(h.ax,'clim',[min(cl) max(cl)]);
    end
    
elseif strcmp(cfg.clim,'sym')
   if cfg.global_clim
       clabs= max(abs(cl - cfg.climzero));
       set(h.ax,'clim',[cfg.climzero-clabs, cfg.climzero+clabs] );
   else
       for ii=1:P
           clabs = max(abs( cl((ii-1)*2+1:ii*2) - cfg.climzero));
           set(h.ax(ii),'clim',[cfg.climzero-clabs, cfg.climzero+clabs] );
       end
   end
end

%% Mark zero line
if ~isempty(cfg.zero)
    if x(1)<0 && x(end)>0
        for ii=1:P
            set(gcf,'CurrentAxes',h.ax(ii));
            hold on
            yl = ylim(gca);
            plot(gca,[0 0], yl(:), cfg.zero{:})
            set(gca,'YLim',yl)
        end
    end
    if y(1)<0 && y(end)>0
        for ii=1:P
            set(gcf,'CurrentAxes',h.ax(ii));
            hold on
            xl = xlim(gca);
            plot(gca,xl(:), [0 0], cfg.zero{:})
            set(gca,'XLim',xl)
        end
    end
end

%% Add colorbar
if cfg.colorbar
    if cfg.global_clim && ~isempty(strfind(lower(cfg.colorbar_location),'outside')) && P>1
        % We place a single colorbar left/above the first image if
        % location='WestOutside'/'NorthOutside', right/below of the last 
        % image if location='EastOutside'/'Southoutside'
        if any(strcmpi(cfg.colorbar_location,{'westoutside','northoutside'}))
            cbpos = 1;
        else
            cbpos = P;
        end
        
%         oldpos = get(h.ax(cbpos),'Position');
        h.colorbar = colorbar('peer',h.ax(cbpos),'location',cfg.colorbar_location);
        set(get(h.colorbar,'title'),'String', cfg.colorbar_title, cfg.colorbar_title_options{:})
%         set(h.ax(cbpos),'Position',oldpos);
    else
        for ii=1:P
            h.colorbar(ii) = colorbar('peer',h.ax(ii),'location',cfg.colorbar_location);
            set(get(h.colorbar(ii),'title'),'String', cfg.colorbar_title, cfg.colorbar_title_options{:})
        end
    end
else
    h.colorbar = [];
end

%% Add labels and title
for ii=1:P  
    if ~isempty(cfg.xlabel{ii}), h.xlabel(ii) = xlabel(h.ax(ii),cfg.xlabel{ii}, cfg.label_options{:}); end
    if ~isempty(cfg.ylabel{ii}), h.ylabel(ii) = ylabel(h.ax(ii),cfg.ylabel{ii}, cfg.label_options{:}); end
    if ~isempty(cfg.title{ii}), h.title(ii) = title(h.ax(ii),cfg.title{ii}, cfg.title_options{:}); end
end

%% Set Y-Dir
set(h.ax,'YDir',cfg.ydir);

%% Add grid
for ii=1:P
    grid(h.ax(ii), cfg.grid{:})
end

%% Set ticks and scale
set(h.ax, 'YLim',[y(1) y(end)]);
set(h.ax, 'XScale',cfg.xscale,'YScale',cfg.yscale);

