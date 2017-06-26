function [dat_out,W,D,A] = mv_ssd_wave(cfg,varargin)
% Wavelet-based implementation of spatio-spectral decomposition (SSD). The
% flanking frequencies are defined at a minimal distance such that their
% bandwidths still do not overlap:
%
% In Morlet wavelets, the frequency bandwidth is given as F/nCycles * 2.
% Thus F - F/nCycles is the left frequency "border" of the wavelet, and 
% F + F/nCycles is the right "border".
% We search left and right flanking frequencies FL and FR such that only
% the "borders" of their bandwidths touch each other:
% FL + FL/nCycles = F - F/nCycles => FL = F * (nCycles-1)/(nCycles+1)
% FR - FR/nCycles = F + F/nCycles => FR = F * (nCycles+1)/(nCycles-1)
%
% Requires Fieldtrip.
%
%Synopsis:
% [dat_out, W, D, A]= MV_SSD_WAVE(CFG,dat,<dat2,dat3,...>)
%
% INPUT:
%     dat  -   data structure of unfiltered continous or epoched data. Data
%              should come in matrix format (not cell array). If multiple
%              datasets are provided (dat2, dat3, etc.) then a common set
%              of components is extracted for all datasets. All datasets
%              should be 2D (continuous) or 3D (epoched).
%
% The configuration should be according to
%
%   foi                 - vector of frequencies to be analysed
%   toi                 - times-of-interest. The
%                         times should be far enough from the epoch border
%                         so that no NaN's occur (consider padding the
%                         signal manually to assure this)
%   nComp               - number of high-SNR SSD components to be retained
%                         (default 10)
%   nCycle              - number of cycles for wavelet (default 6)
%
% OUTPUT:
% dat       - updated data structure with data in the target band
% W         - SSD projection matrix (spatial filters are in the columns)
% D         - generalized eigenvalue score of SSD objective function. This
%             is a measure of the signal-to-noise ratio of the oscillation
% A         - estimated mixing matrix (spatial patterns are in the columns)
%
% Note: For other implementations of SSD in Matlab, check https://github.com/svendaehne/
%
% References:
%
% Nikulin VV, Nolte G, Curio G. A novel method for reliable and fast extraction
% of neuronal EEG/MEG oscillations on the basis of spatio-spectral decomposition.
% NeuroImage, 2011, 55: 1528-1535.
%
% Haufe, S., Dahne, S., & Nikulin, V. V. Dimensionality reduction for the
% analysis of brain oscillations. NeuroImage, 2014
% DOI: 10.1016/j.neuroimage.2014.06.073

% (c) Matthias Treder 2017

dat=varargin;

% Number of datasets
nDat = numel(varargin);
nTrial = zeros(1,nDat);
for dd=1:nDat
    if iscell(dat{dd}.trial)
        error('data %d should come dat matrix format, not as a cell array')
    end
    
    sz= size(dat{dd}.trial);
    if numel(sz)==2
        nTrial(dd)=1;
        nChan = size(dat{dd}.trial,1);
    elseif numel(sz)==3
        nTrial(dd) = sz(1);
        nChan = sz(2);
    else
        error('dat.trial should be 2d or 3d')
    end
end


nFreq = numel(cfg.foi);

mv_setDefault(cfg,'nComp',10);
mv_setDefault(cfg,'toi',[]);
mv_setDefault(cfg,'nCycle',6);

if isempty(cfg.toi)
    error('cfg.toi must be provided')
end
if ~iscell(cfg.toi)
    cfg.toi= {cfg.toi};
end
if numel(cfg.toi)==1 && nDat>1
    cfg.toi = repmat(cfg.toi, [1 nDat]);
end

nComp = cfg.nComp;

%% Prepare outputs
dat_out = repmat({struct()},[1 nDat]);
nTime = zeros(1,nDat);
for dd=1:nDat
    nTime(dd) = numel(cfg.toi{dd});
    if numel(sz)==2
        dat_out{dd}.trial = zeros(nComp,nFreq,nTime(dd));
        dat_out{dd}.dimord = 'chan_freq_time';
    elseif numel(sz)==3
        dat_out{dd}.trial = zeros(nTrial(dd),nComp,nFreq,nTime(dd));
        dat_out{dd}.dimord = 'rpt_chan_freq_time';
    end
    dat_out{dd}.label = arrayfun(@(x) ['SSD' num2str(x)],1:cfg.nComp,'UniformOutput',0);
    if isfield(dat{dd},'trialinfo')
        dat_out{dd}.trialinfo = dat{dd}.trialinfo;
    end
    dat_out{dd}.time = cfg.toi{dd};
end

W= zeros(nChan,nComp,nFreq);
if nargout>2,   D= zeros(nComp,nFreq); end
if nargout>3,   A= zeros(nChan,nComp,nFreq); end

%% Wavelet analysis and perform SSD
for ff=1:nFreq
    F= cfg.foi(ff);
    fprintf('Freq %2.2f Hz\n',F)
    
    % Left and right flanking frequencies
    FL = F * (cfg.nCycle-1)/(cfg.nCycle+1);
    FR = F * (cfg.nCycle+1)/(cfg.nCycle-1);
    
    cfg_wave = [];
    cfg_wave.method     = 'wavelet';
    cfg_wave.width      = cfg.nCycle;
    cfg_wave.output     = 'fourier';
    cfg_wave.foi        = [FL F FR];
    cfg_wave.feedback   = 'no';
    
    %%% ---- Wavelet spectra and covariance ---
    C_s = zeros(nChan);
    C_n = zeros(nChan);
    sig = cell(1,nDat);
    
    for dd=1:nDat %% Loop across dataset and accumulate covariance
        
        % Calculate wavelet spectra
        cfg_wave.toi        = cfg.toi{dd};
        wave = ft_freqanalysis(cfg_wave, dat{dd});
        
        wave = wave.fourierspctrm;
        sig{dd} = squeeze(wave(:,:,2,:));
        wv= real(wave);
        
        % Get covariance matrices of signal and noise
        C_s_tmp = zeros(nChan);
        C_n_tmp = zeros(nChan);
        if numel(sz)==2
            C_s_tmp= nancov(squeeze(wv(:,2,:))');
            C_n_tmp= nancov( squeeze(wv(:,1,:))' + squeeze(wv(:,3,:))');
        else
            for ii=1:nTrial(dd)
                C_s_tmp= C_s_tmp + nancov(squeeze(wv(ii,:,2,:))');
                C_n_tmp= C_n_tmp + nancov( squeeze(wv(ii,:,1,:))' + squeeze(wv(ii,:,3,:))');
            end
            C_s_tmp = C_s_tmp/nTrial(dd);
            C_n_tmp = C_n_tmp/nTrial(dd);
        end
        
        % Add to pooled covariance (across datasets if multiple)
        C_s = C_s + C_s_tmp;
        C_n = C_n + C_n_tmp;
        
    end
    
    % Regularise if necessary
    V = projectIfRankDeficient(sig);
    if nargout > 3
        C= C_s;
    end
    C_s = V' * C_s * V;
    C_n = V' * C_n * V;
    
    
    % ---- SSD ----
    [W_tmp,D_tmp]= eig(C_s,C_s+C_n);
    [ev, sort_idx] = sort(diag(D_tmp), 'descend');
    W_tmp = W_tmp(:,sort_idx);
    W_tmp = V * W_tmp;
    W_tmp = W_tmp(:,1:nComp);
    
    % Project timeseries on SSD components
    for dd=1:nDat
        wave= sig{dd};  % restrict to signal band
        if numel(sz)==2
            dat_out{dd}.trial(:,ff,:) = W_tmp' * wave;
        elseif numel(sz)==3
            for ii=1:nTrial(dd)
                dat_out{dd}.trial(ii,:,ff,:) = W_tmp' * squeeze(wave(ii,:,:));
            end
        end
    end
    
    % assign other output arguments
    if nargout>1
        W(:,:,ff)= W_tmp;
    end
    if nargout>2
        D(:,ff) = ev(1:nComp);
    end
    if nargout > 3
        % A is the matrix with the patterns (in columns)
        A_tmp = C * W_tmp / (W_tmp'* C * W_tmp);
        A(:,:,ff)= A_tmp;
    end
end

for dd=1:nDat
    dat_out{dd}.freq = cfg.foi;
end

if nDat == 1
    dat_out = dat_out{1};
end

%% ----- Helper functions -----
    function [V,C] = projectIfRankDeficient(X)
        % Project data onto subspace if it's not full rank
        % check eigenvalue spectrum
        C= zeros(nChan);
        for ddd=1:nDat
            if numel(sz)==2
                C_tmp= cov(real(X{ddd}));
            elseif numel(sz)==3
                C_tmp= zeros(nChan);
                for jj=1:nTrial(ddd)
                    C_tmp= C_tmp + cov(squeeze(real(X{ddd}(jj,:,:)))');
                end
                C_tmp= C_tmp/nTrial(ddd);
            end
            C = C + C_tmp;
        end
        C_tmp=[];
        
        [V,D_tmp]= eig(C);
        [ev_tmp, sort_idx] = sort(diag(D_tmp), 'descend');
        V = V(:,sort_idx);
        
        % compute an estimate of the rank of the data. If rank deficient, project
        % data on subspace V
        %%% based on github.com/svendaehne/matlab_SSD/blob/master/ssd.m
        tol = ev_tmp(1) * 10^-6;
        r = sum(ev_tmp > tol);
        if r < nChan
            %         fprintf('SSD: Input data does not have full rank. Only %d components can be computed.\n',r);
            V = V(:,1:r) * diag(ev_tmp(1:r).^-0.5);
        else
            V = eye(size(C));
        end
        
    end

end

