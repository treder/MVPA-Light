function [X,y] = simulate_regression_data(typ, nsamples, nfeatures, scale)
% Creates univariate regression data.
%
% Usage:  [X,y] = simulate_regression_data(typ, nsamples, nfeatures, scale)
%
% Parameters:
% typ               - defines the relationship between predictors X and
%                     response variable y. Possible relationships:
%                     'linear':   y = X*w
%                     'sinusoid': y = sin(X*w)
%                     'spiral':   X is a 2-d spiral and y is the Euclidean
%                                 distance to the center
% nsamples          - total number of samples
% nfeatures         - total number of features/predictors
% nclasses          - number of classes (default 2)
% scale             - variance of random Gaussian noise added to y. 
%                     If 0, data is perfectly defined by the functional
%                     relationship. Set to larger valurs to make the data more
%                     fuzzy
%
% Returns:
% X         - [nsamples x nfeatures] matrix of data
% y .       - [nsamples x 1] vector of responses

% (c) Matthias Treder

if nargin<1 || isempty(typ), typ = 'linear'; end
if nargin<2 || isempty(nsamples), nsamples = 100; end
if nargin<3 || isempty(nfeatures), nfeatures = 10; end
if nargin<4 || isempty(scale), scale = 0; end

if strcmp(typ, 'spiral'), nfeatures = 2; end

X = nan(nsamples, nfeatures);
y = nan(nsamples,1);

switch(typ)
    case 'linear'
        %%% --- LINEAR DATA ---
        X = randn(nsamples, nfeatures);
        w = randn(nfeatures, 1);
        y = X*w; 
        
    case 'sinusoid'
        %%% --- SINUSOIDAL DATA ---
        X = randn(nsamples, nfeatures)*100;
        w = randn(nfeatures, 1);
        y = sin(X*w); 
        
    case 'spiral'
        %%% --- SPIRAL DATA ---

        % Start and end phase phase for spiral arm
        spiral_start = rand * 2 * pi;
        nrevolutions = 3;
        spiral_end = spiral_start + nrevolutions * 2 * pi;
        
        % Set phases and distances r for spiral data
        phase = linspace(spiral_start, spiral_end, nsamples)';
        y = linspace(0, 1, nsamples)';
        
        % Convert polar to Cartesian coordinates
        X = [sin(phase).* y, cos(phase).* y];
        
end

%% Add noise to y
y = y + randn(nsamples,1) * scale;
