function [dat,clabel] = load_example_data(filename)

% Load data (in /examples folder)
load(filename)
dat.trial = double(dat.trial);

% attenden_deviant contains the information about the trials. Use this to
% create the true class labels, indicating whether the trial corresponds to
% an attended deviant (1) or an unattended deviant (2).
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;  % Class 1: attended deviants
clabel(~attended_deviant) = 2;  % Class 2: unattended deviants