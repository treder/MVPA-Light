function h = mv_plot_topography(cfg, topo, pos)
%Plots a topography 
%
%Usage:
% ax = mv_plot_topography(cfg, topo, pos)
%
%Parameters:
% topo              - [C x 1] vector with specifying a topography, where C
%                     is the number of channels. 
%                     Alternatively, if a matrix [C x M] is given,
%                     a subplot with M different topographies is created
% pos               - [C x 2] matrix of x-y channel positions. 
%
% cfg          - struct with parameters:
% .outline          - cell array specifying the head/nose/ears cfg.outline. (eg
%                     in Fieldtrip, it corresponds to the lay.cfg.outline
%                     field)  (default [])
% .clim             - set to fix the color limits
%                     [cmin cmax] sets the color limits manually
%                     'minmax' uses the minimum and maximum values in the
%                     plot
%                     'sym' makes the color values symmetric about a
%                     reference point (e.g. [-0.7, 0.7] would be symmetric
%                     about 0. cfg.climzero should be set if 'sym' is selected
%                     (default [])
% .climzero         - the 'zero point' in the data. For classification
%                     accuracy it should correspond to chance level
%                     (default 0.5)
% .globalclim       - if 1 equalises the cfg.clim across all plots (if P>1)
%                     (default 1)
% .colorbar         - if 1, plots a colorbar. If globalclim=1, only one
%                     colorbar is plotted, otherwise there is a separate
%                     colorbar for each topography plotted
% .cbtitle          - title for colorbar
% .title            - string with title. Can be cell array of strings with
%                     if there is multiple topographies eg {'title1'
%                     'title2' 'title3'}
% .mark_chans       - [C x 1] vector with 0's and 1's, where a 1 specifes
%                     that the according channel should be marked.
%                     Alternatively, if multiple topographies are given, 
%                     a matrix [C x M] can be provided
% .res              - resolution of the topography grid in pixel x pixel.
%                     Default 100 gives a 100 x 100 pixel grid
% .nrow, .ncol      - if multiple topographies are given, nrow and ncol
%                     specify the number of rows and columns in the subplot
%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017

if isvector(topo)
    topo = topo(:); % make sure topo is a column vector
end

[~, M] = size(topo);

mv_set_default(cfg,'outline',{});
mv_set_default(cfg,'clim',[]);
mv_set_default(cfg,'climzero', 0.5);
mv_set_default(cfg,'globalclim',0);
mv_set_default(cfg,'colorbar',1);
mv_set_default(cfg,'cbtitle','');
mv_set_default(cfg,'title','');
mv_set_default(cfg,'mark_chans',[]);
mv_set_default(cfg,'res', 100);
mv_set_default(cfg,'ncol',ceil(sqrt(M)));
mv_set_default(cfg,'nrow',ceil(M/cfg.ncol));

if ~iscell(cfg.outline), cfg.outline = {cfg.outline}; end
if ~iscell(cfg.title), cfg.title = {cfg.title}; end

if ~isempty(cfg.mark_chans)
    if isvector(cfg.mark_chans)
        cfg.mark_chans = repmat(cfg.mark_chans(:),[1 M]);
    end
end

h= struct();

%% Data extrapolation

% If an cfg.outline is provided, data can be extrapolated beyond the sensor 
% positions to 'fill up' the scalp cfg.outline. 
% Since griddata can only interpolate (not extrapolate), we add the scalp
% points as extra points with values equal to cfg.climzero.

% Specify x and y limits of the grid
if ~isempty(cfg.outline)
    % assume that the first cell of cfg.outline specifies the scalp cfg.outline
    xminmax = [min(cfg.outline{1}(:,1)), max(cfg.outline{1}(:,1))];
    yminmax = [min(cfg.outline{1}(:,2)), max(cfg.outline{1}(:,2))];
else
    % if no cfg.outline is provided, we just interpolate within the space
    % spanned by the channel positions
    xminmax = [min(pos(:,1)), max(pos(:,1))];
    yminmax = [min(pos(:,2)), max(pos(:,2))];
end

% Create grid
linx = linspace(xminmax(1), xminmax(2), cfg.res);
liny = linspace(yminmax(1), yminmax(2), cfg.res);
[xq,yq] = meshgrid(linx, liny);

% For extrapolation within the cfg.outline, a mask needs to be created that
% masks out everything outside the scalp cfg.outline
if ~isempty(cfg.outline)
    mask = double(inpolygon(xq,yq, cfg.outline{1}(:,1), cfg.outline{1}(:,2)));
    mask(mask==0) = nan;
end

%% Plot
clf
for mm=1:M
    h.ax(mm) = subplot(cfg.nrow,cfg.ncol,mm);
    cla
    
    % Interpolate data 
    if ~isempty(cfg.outline)
        % Use the v4 option which gives us extrapolation as well
        topo_grid = griddata(pos(:,1), pos(:,2), topo(:,mm), xq, yq, 'v4');
        % Mask out the parts outside of the cfg.outline
        topo_grid = topo_grid  .* mask;
    else
        topo_grid = griddata(pos(:,1), pos(:,2), topo(:,mm), xq, yq);
    end
    
    % Plot topography
    contourf(xq, yq, topo_grid,50, 'LineStyle','none');
    hold on
    contour(xq, yq, topo_grid,'LineColor','k','LineStyle','-','LevelList',linspace(min(topo(:,mm)),max(topo(:,mm)),5));
    
    % Color limits
    if ischar(cfg.clim)
        if strcmp(cfg.clim,'sym')
            cl = get(gca,'CLim');
            set(gca,'CLim',[-1,1] * max(abs(cl-cfg.climzero) ) + cfg.climzero )
        end
    elseif ~isempty(cfg.clim)
        set(gca,'Clim',cfg.clim);
    end
    
    % Plot channels
    plot(pos(:,1), pos(:,2), 'k.','MarkerSize',0.5)
    
    % Mark channels
    if ~isempty(cfg.mark_chans)
        plot(pos(cfg.mark_chans(:,mm)==1,1), pos(cfg.mark_chans(:,mm)==1,2), 'k+','MarkerSize',10)
    end
    
    % Plot cfg.outline if given
    if ~isempty(cfg.outline)
        hold on
        for cc=1:numel(cfg.outline)
            lineopt = {'LineWidth', 1, 'Color', 'k'};
            hold on
            plot(cfg.outline{cc}(:,1),cfg.outline{cc}(:,2),lineopt{:})
            % adapt xlim and ylim if necessary
            xl = xlim;
            yl = ylim;
            xlim([min(xl(1),min(cfg.outline{cc}(:,1))), max(xl(2),max(cfg.outline{cc}(:,1)))]);
            ylim([min(yl(1),min(cfg.outline{cc}(:,2))), max(yl(2),max(cfg.outline{cc}(:,2)))]);
        end
    end
    
    % colorbar
    if ~cfg.globalclim && cfg.colorbar 
        h.colorbar(mm) = colorbar('Location','SouthOutside');
        set(get(h.colorbar(mm),'title'),'string',cfg.cbtitle);
    end
    
    % title
    h.title(mm) = title(cfg.title{mod(mm-1,numel(cfg.title))+1},'Interpreter','none');
    
    axis off
end


if cfg.globalclim
    cl = get(h.ax,'CLim');
    cl = cat(1,cl{:});
    cl = [min(cl(:,1)), max(cl(:,2))];
    
    if strcmp(cfg.clim,'sym')
        cl = get(gca,'CLim');
        set(gca,'CLim',[-1,1] * max(abs(cl-cfg.climzero) ) + cfg.climzero )
    else
        set(h.ax,'CLim', cl);
    end
    
    if cfg.colorbar
        if ceil(sqrt(M)) ~= floor(sqrt(M)) 
            % if the subplot grid is not completely filled up, we use the
            % next subplot to produce a colorbar
            h.ax(M+1) = subplot(cfg.nrow,cfg.ncol,M+1);
            h.colorbar = colorbar('Location','WestOutside');
        else
            h.colorbar = colorbar('Location','EastOutside');
        end
        caxis(cl)
        axis off
    end

end