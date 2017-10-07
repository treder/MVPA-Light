function startup_MVPA_Light

% Adds MVPA-Light and its subfolders to the MATLAB path
MVPA_path = fileparts(fileparts(mfilename('fullpath')));

addpath(MVPA_path);
addpath(fullfile(MVPA_path,'classifier'));
addpath(fullfile(MVPA_path,'classifier','optimisation'));
addpath(fullfile(MVPA_path,'examples'));
addpath(fullfile(MVPA_path,'external'));
addpath(fullfile(MVPA_path,'plot'));
addpath(fullfile(MVPA_path,'utils'));
