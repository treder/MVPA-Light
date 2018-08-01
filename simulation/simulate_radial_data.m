function [X,clabel,Y] = simulate_radial_data(nsamples, prop, separation, do_plot)
% Creates two-class radial-shaped data in two dimensions.
% The first class is confined to the unit circle. The second class is
% confined to a ring around the first class. The ring has the same area as
% the unit circle.
%
% Useful for testing non-linear classifiers eg SVM with RBF kernel.
%
% Usage:  [X,clabel,Y,M] = simulate_radial_data(nsamples, , nclasses, prop, scale, do_plot)
%
% Parameters:
% nsamples          - total number of samples (across all classes)
% prop              - class proportions (default 'equal'). Otherwise give
%                     class proportions e.g. [0.2, 0.8] gives 20% of
%                     the samples to class 1, 80% to class 2.
% separation        - distance between the unit circle and the ring
% do_plot           - if 1, plots the data 
%
% Returns:
% X         - [nsamples x 2] matrix of data
% clabel    - [nsamples x 1] vector of class labels (1's, 2's, etc)
% Y         - [samples x nclasses] indicator matrix that can be used
%             alongside or instead of clabel. Y(i,j)=1 if the i-th sample
%             belongs to class j, and Y(i,j)=0 otherwise.

% (c) Matthias Treder

if nargin<2 || isempty(prop), prop = 'equal'; end
if nargin<3 || isempty(separation), separation = 0; end
if nargin<4 || isempty(do_plot), do_plot = 1; end

% Check input arguments
if ischar(prop) && strcmp(prop,'equal') && ~(rem(nsamples, 2)==0)
    error('Class proportion is set to ''equal'' but number of samples cannot be divided by 2')
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

%% Determine frequencies of each class
if ischar(prop) && strcmp(prop,'equal')
    nsamples_per_class = nsamples / 2 * ones(1, 2);
else
    nsamples_per_class = nsamples * prop;
end

if ~all(mod(nsamples_per_class,1)==0)
    error('prop * nsamples must yield integer values')
end


%% Generate 'inner' class 1

% Create samples using polar coordinates: radius [r] and phase [phi]
r = rand(nsamples_per_class(1),1);
phi = rand(nsamples_per_class(1),1) * 2 * pi;

% Convert into Cartesian coordinates
X(1:nsamples_per_class(1), :) = [r .* cos(phi), r .* sin(phi)];

%% Generate 'outer' class 2

% Start radios: add separation
r_start = 1 + separation;

% End radius such that area of ring = pi
r_end = sqrt(1 + r_start^2);

% Create samples using polar coordinates
r = rand(nsamples_per_class(2),1) * (r_end-r_start) + r_start;
phi = rand(nsamples_per_class(2),1) * 2 * pi;

% Convert into Cartesian coordinates
X(nsamples_per_class(1)+1:end, :) = [r .* cos(phi), r .* sin(phi)];

%% Clabel
clabel = [ones(nsamples_per_class(1),1); 2*ones(nsamples_per_class(2),1)];

%% Plot data
if do_plot
    
    plotopt = {'.', 'MarkerSize', 12};
    
    clf
    for cc=1:2
        plot(X(clabel==cc,1),X(clabel==cc,2), plotopt{:})
        hold all
    end
    xlabel('Feature 1')
    ylabel('Feature 2')
  
end
