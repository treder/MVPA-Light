%%% This examples explores searchlight classification: classification is 
%%% repeated for each electrode separately. Since we then obtain a 
%%% classification performance value for each electrode, we can plot the 
%%% performance as a scalp topography. 
%%%
%%% In addition to considering each electrode in isolation we can also 
%%% consider an electrode and its closest neighbours - the number of
%%% closest neighbours that we consider then determines the size of the 
%%% 'searchlight'
%%%
clear

% Load data (in /examples folder)
[dat, clabel, chans] = load_example_data('epoched3');

% We want to classify on the 300-500 ms window
time_idx = find(dat.time >= 0.3  &  dat.time <= 0.5);

%% Distance matrix giving the pair-wise distances between electrodes
nb_mat = squareform(pdist(chans.pos));

%% Searchlight classification
cfg = [];
cfg.neighbours  = nb_mat;
cfg.average     = 1;
cfg.metric      = 'auc';
cfg.size        = 3;    % consider each electrode and its 3 closest neighbouring electrodes

[auc, result] = mv_searchlight(cfg, dat.trial(:,:,time_idx), clabel);

%% Plot classification performance as a topography
cfg_plot = [];
cfg_plot.outline = chans.outline;
figure
mv_plot_topography(cfg_plot, auc, chans.pos);
colormap jet
title(sprintf('Searchlight [%d neighbours]', cfg.size))

%% Repeat the searchlight analysis for different numbers of neighbours
perf = cell(1,6);
for nn=0:5

    fprintf('\n----- Searchlight [%d neighbours] -----\n', cfg.size)
    
    % Vary the number of neighbours
    cfg.size    = nn;
    
    perf{nn+1} = mv_searchlight(cfg, dat.trial(:,:,time_idx), clabel);
    
end

%% -- end of example --
return

%% Alternative approach for defining the neighbours  using Fieldtrip
%%% Note: This example requires FieldTrip, since we define the
%%% neighbourhood structure using a FieldTrip function. 
%
% We use Fieldtrip to obtain the label layout and the neighbours. The
% neighborhood matrix is defined as a graph here consisting of 1's for
% neighbouring electrodes and 0's for non-neighbouring ones
cfg = [];
% cfg.method      = 'triangulation';  %'distance'
cfg.method      = 'distance';
cfg.neighbourdist = 0.195;
cfg.layout      = 'easycapM1';
cfg.feedback    = 'yes';
cfg.channel     = dat.label;
neighbours= ft_prepare_neighbours(cfg);

nChan = numel(dat.label);

% Create neighbours matrix
nb_mat = zeros(nChan);

for ii=1:nChan
    
    % Find index of current channel in neighbours array
    idx = find(ismember({neighbours.label},dat.label{ii}));
    
    % Find indices of its neighbours in dat.label
    idx_nb = find(ismember(dat.label, neighbours(idx).neighblabel))';
    
    % We only take 2 neighbours
    nb_mat(ii,[ii, idx_nb]) = 1;

end

figure,
imagesc(nb_mat)
set(gca,'XTickLabel',dat.label(get(gca,'XTick')))
set(gca,'YTickLabel',dat.label(get(gca,'YTick')))
title('Neighbourhood matrix')
grid on

%% Searchlight analysis
cfg = [];
cfg.neighbours  = nb_mat;
cfg.average     = 1;

maxstep = 2;        % maximum neighbourhood size
auc = cell(1,maxstep);

%%% Start classification 
%%% - In the first iteration, size = 0, i.e. only an electrode alone is
%%%   considered
%%% - In the second iteration, size = 1, and an electrode as well as its
%%%   direct neighbours are considered
%%% - In the third iteration, size = 2, so an electrode, its direct
%%%   neighbours, and the neighbours of the neighbours are considered
for size=0:maxstep
    cfg.size  = size;
    auc{size+1} = mv_searchlight(cfg, dat.trial(:,:,time_idx), clabel);
end
