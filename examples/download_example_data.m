function download_example_data(dataset, target_dir)
%Downloads datasets used in the examples. 
%
%Usage:
%  download_example_data(dataset, target_dir)
%
%Parameters:
% dataset            - name of the dataset. Possible values: 'music_bci'
% target_dir         - target directory for the data. By default, the data
%                      is saved in the MVPA Light examples folder.
%

if nargin < 2
    [target_dir, ~] = fileparts(which(mfilename)); 
end

switch(dataset)
    case 'music_bci'
        % see: http://bnci-horizon-2020.eu/database/data-sets
        subjects = {'VPaak' 'VPaan' 'VPgcc' 'VPaap' 'VPaaq' 'VPjaq' 'VPaar' 'VPjat' 'VPgeo' 'VPaas' 'VPaat'};
        url = 'http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-MusicBCI/'; 
        fprintf('Downloading %d files from %s\n', numel(subjects), url)
        fprintf('NOTE: The files are large (~600 MB each file) so the download might take a long time\n')
        filename = sprintf('musicbci_%s.mat', subject);
    otherwise
        error('Unknown dataset: %s', dataset)
end