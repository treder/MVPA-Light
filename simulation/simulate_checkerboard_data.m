function [X,clabel,Y] = simulate_checkerboard_data(nsamples, nchecker, nclasses, prop, scale, do_plot)
% Creates checkerboard data in two dimensions. The space is split into an
% nchecker x nchecker checkerboard. Each checker contains data from one
% single class.
%
% Note that checkers are filled row-by-row, cycling through each of the
% classes. Therefore, ideally the number of checkers is nclasses+1.
%
% Usage:  [X,clabel,Y,M] = simulate_checkerboard_data(nsamples, nchecker, nclasses, prop, scale, do_plot)
%
% Parameters:
% nsamples          - total number of samples (across all classes)
% nchecker          - number of checkers (default 3)
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

if nargin<2 || isempty(nchecker), nchecker = 3; end
if nargin<3 || isempty(nclasses), nclasses = 2; end
if nargin<4 || isempty(prop), prop = 'equal'; end
if nargin<5 || isempty(scale), scale = 0; end
if nargin<6 || isempty(do_plot), do_plot = 1; end

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

%% Determine frequencies of each class
if ischar(prop) && strcmp(prop,'equal')
    nsamples_per_class = nsamples / nclasses * ones(1, nclasses);
else
    nsamples_per_class = nsamples * prop;
end

if ~all(mod(nsamples_per_class,1)==0)
    error('prop * nsamples must yield integer values')
end

% How many checkers each class occupies
total_checkers = nchecker^2;
checker_classes = repmat(1:3, 1, ceil(total_checkers/nclasses));
checker_classes = checker_classes(1:total_checkers);
checkers_per_class = arrayfun( @(c) sum(checker_classes==c), 1:nclasses);

%% Loop across checkers
X = [];
clabel = [];

x = 0; y = 0; 
current_checker_position = 1;

% Count how many samples per class have already be added
nsamples_per_class_added = zeros(1,nclasses);

for rr=1:nchecker       % -- rows
    for cc=1:nchecker     % -- cols
        % Current checker class
        ccc = checker_classes(current_checker_position);
        
        % how many samples to generate
        if sum(checker_classes(current_checker_position:end)==ccc) == 1
            % this is the last checker of this class, we need to add all
            % missing samples
            nadd = nsamples_per_class(ccc) - nsamples_per_class_added(ccc);
        else
            % if the nr of samples-per-class is
            % not divisible by the number of checkers-per-class, we have to
            % distribute the odd ones
            
            if mod(sum(checker_classes(1:current_checker_position)),2)==0
                nadd = floor(nsamples_per_class(ccc) / checkers_per_class(ccc));
            else
                nadd = ceil(nsamples_per_class(ccc) / checkers_per_class(ccc));
            end
            nsamples_per_class_added(ccc) = nsamples_per_class_added(ccc) + nadd;
        end
        
        % create data within this checker
        X = [X; rand(nadd, 2)+repmat([x,y],nadd,1) ];
        clabel = [clabel; ones(nadd,1) * ccc];
        
        % increase count
        x = x + 1;
        current_checker_position = current_checker_position + 1;
    end
    x = 0;
    y = y + 1;
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
