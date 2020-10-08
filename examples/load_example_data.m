function [dat,clabel, chans] = load_example_data(filename, do_zscore)
% Loads an example dataset. 
%
% filename  - epoched1.mat, epoched2.mat, epoched3.mat
%
% Output: 
% dat       - struct containing the data
% clabel    - class labels
% chans     - struct specifying the positions of the channels for plotting

if nargin<2, do_zscore = 1; end

%%% DATA 

% Load data (in /examples folder)
load(filename)
dat.trial = double(dat.trial);

% For logistic regression, it is important that the data are scaled well.
% We therefore apply z-scoring per default.
if do_zscore
    dat.trial = zscore(dat.trial,[],1);
end

%%% CLASS LABELS 

% attended_deviant contains the information about the trials. Use this to
% create the true class labels, indicating whether the trial corresponds to
% an attended deviant (1) or an unattended deviant (2).
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;  % Class 1: attended deviants
clabel(~attended_deviant) = 2;  % Class 2: unattended deviants

