% create the channel layout and exports it together with the example files
clear 

for fname = {'epoched1' 'epoched2' 'epoched3'}
   
    load(fname{:});
    
    % Create layout
    cfg.layout  = 'EasycapM1';
    cfg.channel = dat.label;
    chans = ft_prepare_layout(cfg);
    
    % Reorder the channels such that they appear in the same order as in
    % dat.label
    reord = nan(numel(dat.label),1);
    for ii=1:numel(dat.label)
        reord(ii) = find(ismember(chans.label, dat.label{ii}));
    end
    
    chans.label = chans.label(reord);
    chans.pos = chans.pos(reord,:);
    chans.width = chans.width(reord);
    chans.height = chans.height(reord);
    
    % save - take care that you are in the /examples subfolder
    save(fname{:}, 'dat','attended_deviant','chans','nChan','nTime','nTrial');
    
end

fprintf('Finished all\n')