function [X,clabel,Y] = simulate_spiral_data(nsamples, nrevolutions, nclasses, prop, scale, do_plot)
% Creates spiral-shaped data in two dimensions.
% Useful for testing non-linear classifiers eg SVM with RBF kernel.
%
% Usage:  [X,clabel,Y,M] = simulate_spiral_data(nsamples, nrevolutions, nclasses, prop, scale, do_plot)
%
% Parameters:
% nsamples          - total number of samples (across all classes)
% nrevolutions      - number of 360 deg revolutions each spiral arm takes
%                     (default 1/4)
% nclasses          - number of classes (default 2)
% prop              - class proportions (default 'equal'). Otherwise give
%                     class proportions e.g. [0.2, 0.2, 0.6] gives 20% of
%                     the samples to class 1, 20% to class 2, and 60% to
%                     class 3
% scale             - variance of random Gaussian noise added to the x-y-
%                     positions. If 0, data is perfectly arranged on a
%                     line. Set to larger valurs to make the data more
%                     fuzzy
% do_plot           - if 1, plots the data 
%
% Returns:
% X         - [nsamples x 2] matrix of data
% clabel    - [nsamples x 1] vector of class labels (1's, 2's, etc)
% Y         - [samples x nclasses] indicator matrix that can be used
%             alongside or instead of clabel. Y(i,j)=1 if the i-th sample
%             belongs to class j, and Y(i,j)=0 otherwise.

% (c) Matthias Treder

if nargin<2 || isempty(nrevolutions), nrevolutions = 1/4; end
if nargin<3 || isempty(nclasses), nclasses = 2; end
if nargin<4 || isempty(prop), prop = 'equal'; end
if nargin<5 || isempty(scale), scale = 0; end
if nargin<6 || isempty(do_plot), do_plot = 0; end

% Check input arguments
if ischar(prop) && strcmp(prop,'equal') && ~(rem(nsamples, nclasses)==0)
    error('Class proportion is set to ''equal'' but number of samples cannot be divided by the number of classes')
end

if ~ischar(prop)
    if sum(prop) ~= 1
        error('prop must sum to 1')
    end
    if numel(prop) ~= nclasses
        error('number of elements in prop must match nclasses')
    end
end


X = nan(nsamples, 2);
clabel = nan(nsamples,1);
Y = zeros(nsamples,nclasses);

%% Determine frequencies of each class
if ischar(prop) && strcmp(prop,'equal')
    nsamples_per_class = nsamples / nclasses * ones(1, nclasses);
else
    nsamples_per_class = nsamples * prop;
end

if ~all(mod(nsamples_per_class,1)==0)
    error('prop * nsamples must yield integer values')
end

%% Start phase for each of the spiral arms
spiral_start = linspace(0, 2*pi, nclasses+1);
spiral_start = spiral_start(1:end-1);

%% End phase for each of the spiral arms
spiral_end = spiral_start + nrevolutions * 2 * pi;

%% Generate data and class labels
n = 1;

for cc=1:nclasses
    
    % Set phases and distances r for spiral data
    phase = linspace(spiral_start(cc), spiral_end(cc), nsamples_per_class(cc))';
    r = linspace(0, 1, nsamples_per_class(cc))';
    
    % Convert polar to Cartesian coordinates
    X(n:n+nsamples_per_class(cc)-1,:) = [sin(phase).*r, cos(phase).*r];
    
    % Set labels
    clabel(n:n+nsamples_per_class(cc)-1) = cc;
    
    % Set indicator matrix
    Y(n:n+nsamples_per_class(cc)-1, cc) = 1;
    
    n = n+nsamples_per_class(cc);
end

%% Add noise to x-y positions
if scale > 0
    X = X + randn(size(X)) * sqrt(scale);
end

%% Plot data
if do_plot
    
    plotopt = {'.', 'MarkerSize', 12};
    
    clf
    for cc=1:nclasses
        plot(X(clabel==cc,1),X(clabel==cc,2), plotopt{:})
        hold all
    end
    xlabel('Feature 1')
    ylabel('Feature 2')
  
end
