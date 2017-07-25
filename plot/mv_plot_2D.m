function h = mv_plot_2D(cfg, dat)
%Plots 2D results, e.g. a time x time generalisation. Plots the results as
%a color matrix. If a 3D matrix is given several subplots are created.
%
%Usage:
% ax = mv_plot_2D(cfg,M)
%
%Parameters:
% DAT               - [N x M] data matrix with results or [N x M x P] 3D
%                     matrix with P different images
%
% cfg          - struct with hyperparameters:
% xlabel,ylabel     - label for x and y axes (default 'Training time' and
%                     'Testing time')
% title             - axis title (default '')
% x, y              - x and y values (e.g. time points, frequencies). the 
%                     number of elements in x and y should match the size of DAT
% grid              - options for the grid function (default {'on'})
% clim              - set to fix the color limits
%                     [cmin cmax] sets the color limits manually
%                     'minmax' uses the minimum and maximum values in the
%                     plot
%                     'sym' makes the color values symmetric about a
%                     reference point (e.g. [-0.7, 0.7] would be symmetric
%                     about 0. climzero should be set if 'sym' is selected
%                     (default 'minmax')
% climzero          - if clim = 'sym' a "zero-point" needs to be defined
%                     (for instance 0.5 for classification accuracies)
%                     (default 0)
% globalclim        - if 1 equalises the clim across all plots (if P>1)
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
%
% Specific options for the COLORBAR:
% colorbar          - if 1 a colorbar is added to each plot. If there is
%                     multiple plots and globalclim=1, a single colorbar is
%                     plotted if the colorbar location is outside (default 1)
% cblocation        - colorbar location, see help colorbar (default
%                     'EastOutside')

%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017

[nX,nY,P] = size(dat);

mv_setDefault(cfg,'x',1:nX);
mv_setDefault(cfg,'y',1:nY);
mv_setDefault(cfg,'xlabel','Testing time');
mv_setDefault(cfg,'ylabel','Training time');
mv_setDefault(cfg,'title','');
mv_setDefault(cfg,'clim','maxmin');
mv_setDefault(cfg,'climzero',0);
mv_setDefault(cfg,'globalclim',1);
mv_setDefault(cfg,'grid',{'on'});
mv_setDefault(cfg,'zero',{'--k'});
mv_setDefault(cfg,'nrow',1);
mv_setDefault(cfg,'ncol',P);
mv_setDefault(cfg,'colorbar',1);
mv_setDefault(cfg,'cblocation','EastOutside');
mv_setDefault(cfg,'ydir','normal');
mv_setDefault(cfg,'yscale','linear');
mv_setDefault(cfg,'xscale','linear');

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

x= cfg.x;
y= cfg.y;

%% Plot data matrix
axnum=1;
for rr=1:cfg.nrow
    for cc=1:cfg.ncol
        if axnum<=P
            h.ax(axnum) = subplot(cfg.nrow,cfg.ncol,axnum);
            set(gcf,'CurrentAxes',h.ax(axnum));
            
            % Plot the classification performance image here. The y-axis
            % represents training time and the x-axis represents testing
            % time
            imagesc(cfg.x, cfg.y, squeeze(dat(:,:,axnum)));
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
    if cfg.globalclim
        set(h.ax,'clim',[min(cl) max(cl)]);
    end
    
elseif strcmp(cfg.clim,'sym')
   if cfg.globalclim
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
            set(gca,'YLim',yl)
        end
    end
end

%% Add colorbar
if cfg.colorbar
    if cfg.globalclim && ~isempty(strfind(lower(cfg.cblocation),'outside'))
        % We place a single colorbar left/above the first image if
        % location='WestOutside'/'NorthOutside', right/below of the last 
        % image if location='EastOutside'/'Southoutside'
        if any(strcmpi(cfg.cblocation,{'westoutside','northoutside'}))
            cbpos = 1;
        else
            cbpos = P;
        end
        
        oldpos = get(h.ax(cbpos),'Position');
        cb = colorbar('peer',h.ax(cbpos),'location',cfg.cblocation);
        h.colorbar = cb;
        set(h.ax(cbpos),'Position',oldpos);
    else
        for ii=1:P
            h.colorbar(ii) = colorbar('peer',h.ax(ii),'location',cfg.cblocation);
        end
    end
end

%% Add labels and title
for ii=1:P  
    if ~isempty(cfg.xlabel{ii}), h.xlabel(ii) = xlabel(h.ax(ii),cfg.xlabel{ii}); end
    if ~isempty(cfg.ylabel{ii}), h.ylabel(ii) = ylabel(h.ax(ii),cfg.ylabel{ii}); end
    if ~isempty(cfg.title{ii}), h.title(ii) = title(h.ax(ii),cfg.title{ii}); end
end

%% Set Y-Dir
set(h.ax,'YDir',cfg.ydir);

%% Add grid
for ii=1:P
    grid(h.ax(ii), cfg.grid{:})
end

%% Set ticks and scale
set(h.ax, 'YLim',[cfg.y(1) cfg.y(end)]);
set(h.ax, 'XScale',cfg.xscale,'YScale',cfg.yscale);

