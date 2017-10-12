function h = mv_plot_topography(topo, pos, outline, clim, climzero, globalclim, res)
%Plots a topography 
%
%Usage:
% ax = mv_plot_topography(topo)
%
%Parameters:
% topo              - [C x 1] vector with specifying a topography, where C
%                     is the number of channels. 
%                     Alternatively, if a matrix [C x M] is given,
%                     a subplot with M different topographies is created
% pos               - [C x 2] matrix of x-y channel positions. 
% outline           - cell array specifying the head/nose/ears outline. (eg
%                     in Fieldtrip, it corresponds to the lay.outline
%                     field)  (default [])
% clim              - set to fix the color limits
%                     [cmin cmax] sets the color limits manually
%                     'minmax' uses the minimum and maximum values in the
%                     plot
%                     'sym' makes the color values symmetric about a
%                     reference point (e.g. [-0.7, 0.7] would be symmetric
%                     about 0. climzero should be set if 'sym' is selected
%                     (default [])
% climzero          - the 'zero point' in the data. For classification
%                     accuracy it should correspond to chance level
%                     (default 0.5)
% globalclim        - if 1 equalises the clim across all plots (if P>1)
%                     (default 1)
% res               - resolution of the topography grid in pixel x pixel.
%                     Default 100 gives a 100 x 100 pixel grid
%
% Returns:
% h        - struct with handles to the graphical elements 

% (c) Matthias Treder 2017

if nargin<3 || isempty(outline),    outline = {}; end
if nargin<4 || isempty(clim),       clim = []; end
if nargin<5 || isempty(climzero), 	climzero = 0.5; end
if nargin<6 || isempty(globalclim), globalclim = 0; end
if nargin<7 || isempty(res),        res = 100; end

if ~iscell(outline)
    outline = {outline};
end

if isvector(topo)
    topo = topo(:); % make sure topo is a column vector
end

[~, M] = size(topo);

h= struct();

%% Data extrapolation

% If an outline is provided, data can be extrapolated beyond the sensor 
% positions to 'fill up' the scalp outline. 
% Since griddata can only interpolate (not extrapolate), we add the scalp
% points as extra points with values equal to climzero.
extrax = [];
extray = [];
extravals = [];

% Specify x and y limits of the grid
if ~isempty(outline)
    % assume that the first cell of outline specifies the scalp outline
    xminmax = [min(outline{1}(:,1)), max(outline{1}(:,1))];
    yminmax = [min(outline{1}(:,2)), max(outline{1}(:,2))];
else
    % if no outline is provided, we just interpolate within the space
    % spanned by the channel positions
    xminmax = [min(pos(:,1)), max(pos(:,1))];
    yminmax = [min(pos(:,2)), max(pos(:,2))];
end

% Create grid
linx = linspace(xminmax(1), xminmax(2), res);
liny = linspace(yminmax(1), yminmax(2), res);
[xq,yq] = meshgrid(linx, liny);

% For extrapolation within the outline, a mask needs to be created that
% masks out everything outside the scalp outline
if ~isempty(outline)
    mask = double(inpolygon(xq,yq, outline{1}(:,1), outline{1}(:,2)));
    mask(mask==0) = nan;
end

%% Plot

for mm=1:M
    h.ax(mm) = subplot(1,M,mm);
    cla
    
    % Interpolate data 
    if ~isempty(outline)
        % Use the v4 option which gives us extrapolation as well
        topo_grid = griddata(pos(:,1), pos(:,2), topo(:,mm), xq, yq, 'v4');
        % Mask out the parts outside of the outline
        topo_grid = topo_grid  .* mask;
    else
        topo_grid = griddata(pos(:,1), pos(:,2), topo(:,mm), xq, yq);
    end
    % Plot topography
    contourf(xq, yq, topo_grid,50, 'LineStyle','none');
    hold on
    contour(xq, yq, topo_grid,'LineColor','k','LineStyle','-','LevelList',linspace(min(topo(:,mm)),max(topo(:,mm)),5));
    
    % Color limits
    if ischar(clim)
        if strcmp(clim,'sym')
            
        elseif strcmp(clim,'minmax')
        end
    elseif ~isempty(clim)
        set(gca,'CLim',clim);
    end
    
    % Plot channels
    plot(pos(:,1), pos(:,2), 'k.')
    
    % Plot outline if given
    if ~isempty(outline)
        hold on
        for cc=1:numel(outline)
            lineopt = {'LineWidth', 1, 'Color', 'k'};
            plot(outline{cc}(:,1),outline{cc}(:,2),lineopt{:})
        end
    end
    
    axis off
end

if globalclim
end