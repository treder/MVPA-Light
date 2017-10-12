%%% Perform searchlight classification: classification is repeated for each
%%% electrode separately. As a result, we can plot classification
%%% performance as a topography. 
%%%
%%% In addition to considering each electrode in isolution we also consider
%%% an electrode and its direct neighbours - this is the 'searchlight'
%%%
%%% Note: This example requires FieldTrip, since we define the
%%% neighbourhood structure using a FieldTrip function. If you do not use
%%% FieldTrip, you can replace the next cell with your own code for
%%% building a neighbourhood matrix and then run the rest of the code.

[dat, clabel, chans] = load_example_data('epoched3');

nChan = numel(dat.label);

% We want to classify focus on the 300-500 ms window
time_idx = find(dat.time >= 0.3  &  dat.time <= 0.5);

%% Distance matrix giving the pair-wise distances between electrodes
nb_mat = squareform(pdist(chans.pos));

%% Searchlight analysis
cfg = [];
cfg.nb          = nb_mat;
cfg.average     = 1;

maxstep = 2;        % maximum neighbourhood size
auc = cell(1,maxstep);

%%% Start classification 
%%% - In the first iteration, nbstep = 0, i.e. only an electrode alone is
%%%   considered
%%% - In the second iteration, nbstep = 1, and an electrode as well as its
%%%   direct neighbours are considered
%%% - In the third iteration, nbstep = 2, so an electrode, its direct
%%%   neighbours, and the neighbours of the neighbours are considered
for nbstep=0:maxstep
    cfg.nbstep  = nbstep;
%     cfg.max     = nbstep + 2;
    auc{nbstep+1} = mv_searchlight(cfg, dat.trial(:,:,time_idx), clabel);
end

%% Plot classification performance as a topography [requires Fieldtrip]

% Plot topography and electrode layout
clf
nRow= 1; nCol = maxstep+1;
for ii=1:maxstep+1
    subplot(nRow,nCol,ii)
    
    ft_plot_topo(lay.pos(:,1), lay.pos(:,2), auc{ii}, ...
        'mask',lay.mask,'datmask',[],'interplim','mask');
    ft_plot_lay(lay,'box','no','label','no','point','yes','pointsymbol','o','pointcolor','k','pointsize',4)
    
    colorbar('location','southoutside')
    title(sprintf('nbstep = %d',ii-1))
    colormap jet
end


%% -- end of example --

%% Alternative approach for defining the neighbours  using Fieldtrip
% Here, we use Fieldtrip to obtain the label layout and the neighbours. The
% neighborhood matrix is defined as a graph here consisting of 1's for
% neighbouring electrodes and 0's for non-neighbouring ones
cfg = [];
% cfg.method      = 'triangulation';  %'distance'
cfg.method      = 'distance';
cfg.neighbourdist = 0.195;
cfg.layout      = 'EasycapM1';
cfg.feedback    = 'yes';
cfg.channel     = dat.label;
neighbours= ft_prepare_neighbours(cfg);

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
cfg.nb          = nb_mat;
cfg.average     = 1;

maxstep = 2;        % maximum neighbourhood size
auc = cell(1,maxstep);

%%% Start classification 
%%% - In the first iteration, nbstep = 0, i.e. only an electrode alone is
%%%   considered
%%% - In the second iteration, nbstep = 1, and an electrode as well as its
%%%   direct neighbours are considered
%%% - In the third iteration, nbstep = 2, so an electrode, its direct
%%%   neighbours, and the neighbours of the neighbours are considered
for nbstep=0:maxstep
    cfg.nbstep  = nbstep;
%     cfg.max     = nbstep + 2;
    auc{nbstep+1} = mv_searchlight(cfg, dat.trial(:,:,time_idx), clabel);
end
