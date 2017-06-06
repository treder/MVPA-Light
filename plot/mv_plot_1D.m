%% TODO 
function h = mv_plot_1D(cfg, dat)
%Plots 1D results, e.g. classification across time. If a 2D matrix is given 
%several plots are created.
%
%Usage:
% ax = mv_plot_2D(cfg,M)
%
%Parameters:
% DAT               - [N x M] data matrix with results. Plots M lines of
%                     length M
%
% cfg          - struct with hyperparameters:
% xlabel,ylabel     - label for x and y axes (default '')
% title             - axis title (default '')
% x                 - x values (e.g. time points)
% grid              - options for the grid function (default {'on'})
% zero              - marks the zero point with a horizontal and vertical
%                     line. Give the line options as cell array (default 
%                     {'--k'}). Set to [] to remove lines
%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017

[nX,nY] = size(dat);

mv_setDefault(cfg,'x',1:nX);
mv_setDefault(cfg,'xlabel','');
mv_setDefault(cfg,'ylabel','');
mv_setDefault(cfg,'title','');
mv_setDefault(cfg,'grid',{'on'});
mv_setDefault(cfg,'zero',{'--k'});


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

%% Plot data matrix
axnum=1;
for rr=1:cfg.nrow
    for cc=1:cfg.ncol
        if axnum<=P
            h.ax(axnum) = subplot(cfg.nrow,cfg.ncol,axnum);
            set(gcf,'CurrentAxes',h.ax(axnum));
            % Paint the image
            imagesc(cfg.x, cfg.y, squeeze(dat(:,:,axnum)));
            axnum=axnum+1;
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

