function [pparam, K, clabel] = mv_preprocess_average_kernel(pparam, K, clabel)
% Kernel averaging is the generalization of sample averaging to the
% non-linear case. The samples are not averaged in input space (which is
% non-linear) but rather in Reproducing Kernel Hilbert Space (RKHS), in 
% which the data is linearly separable. The approach was introduced in
% Treder (2018).
%
% For linear classification problems, the kernel averaging approach is
% equivalent to sample averaging.
%
% Note: input data must be a precomputed kernel matrix.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_average_kernel(pparam, X, clabel)
%
%Parameters:
% X              - [samples x samples x ...] kernel matrix
% clabel         - [samples x 1] vector of class labels
%
% pparam         - [struct] with preprocessing parameters
% .group_size       - group size, ie number of samples per group. If the
%                     total number of samples is not divisible by group_size, 
%                     some samples are discarded (default 5)
%
% References:
% Cichy, R.M., Pantazis, D. (2017): Multivariate pattern analysis of MEG and EEG: 
% a comparison of representational structure in time and space. NeuroImage 
% 158, 441–454. https://doi.org/10.1016/j.neuroimage.2017.07.0235.
%
% Treder, M. S. (2018). Improving SNR and reducing training time of 
% classifiers in large datasets via kernel averaging. Lecture Notes in 
% Computer Science, 11309, 239–248. https://doi.org/10.1007/978-3-030-05587-5_23

nclasses = max(clabel);
sz = size(K);

% indices of samples per class
indices = arrayfun(@(c) find(clabel == c), 1:nclasses, 'Un', 0);

% number of samples per class
npc = cellfun(@numel, indices);

% number of averages per class
nav = floor(npc / pparam.group_size);

% create groups of samples as a nested cell array containing [classes x
% groups]
randix = cell(nclasses, 1);
for cc=1:nclasses   % -- loop across classes
    % within each class, create nested cell array with indices for the
    % groups
    randix{cc} = cell(nav(cc),1);

    % randomly shuffle indices in a class
    tmpix = indices{cc}(randperm(npc(cc)));
    for aa=1:nav(cc)    % -- loop across averages within classes
        randix{cc}{aa} = tmpix((aa-1)*pparam.group_size+1 : aa*pparam.group_size);
    end
end

% Create class labels for averaged data
clabel = arrayfun(@(c) ones(nav(c),1)*c, 1:nclasses, 'Un',0);
clabel = cat(1, clabel{:});


% KERNEL AVERAGING
% In the kernel case we need to average both rows and cols of the kernel
% matrix. But the approach is different in the train and test phases:
% (1)   train phase: requires a [train samples x train samples] matrix,
%       hence we average both rows/cols equally
% (2)   test phase: requires a [test samples x train samples] matrix,
%       hence we use the current randix to average the test rows but we
%       need the randix from the train phase to average the cols

% we use s to variably select either rows or cols
s = repmat({':'}, [1, ndims(K)]);
for av_dim=1:2  % -- loop across dimensions to be averaged
    
    if av_dim==2 && pparam.is_train_set == 0
        % need to take the indices from the train phase
        nclasses = pparam.nclasses;
        randix   = pparam.randix;
        nav      = pparam.nav;
    end
    
    % set size of dimension to number of averages
    sz(av_dim) = sum(nav);
    tmp = zeros(sz);
    pos = 1;
    s_group = s;
    
    % Average X
    for cc=1:nclasses     % -- loop across classes
        for aa=1:nav(cc)    % -- loop across averages within classes
            s_group(av_dim) = randix{cc}(aa);
            
            % Extract samples for current group
            Xtmp = K(s_group{:});
            
            % Store result in X
            s_new = s;
            s_new(av_dim) = {pos};
            tmp(s_new{:}) = squeeze(mean(Xtmp, av_dim));
            pos = pos + 1;
        end
    end
    
    % Store averaged version back in X
    K = tmp;
end

if pparam.is_train_set
    pparam.nclasses = nclasses;
    pparam.randix   = randix;
    pparam.nav      = nav;
end

% % there can be multiple sample dimensions. Therefore, we build a colon
% % operator to extract the train/test samples irrespective of the
% % position and number of sample dimensions
% s = repmat({':'},[1, ndims(X)]);
% 
% % since there can be multiple sample dimensions, we average the data matrix
% % dimension-by-dimension. In each iteration, one dimension is averaged and
% % the result is being stored in tmp
% for av_dim=sd  % -- loop across dimensions to be averaged
%     
%     % set size of dimension to number of averages
%     sz(av_dim) = sum(nav);
%     tmp = zeros(sz);
%     pos = 1;
%     
%     % Average X
%     for cc=1:nclasses     % -- loop across classes
%         for aa=1:nav(cc)    % -- loop across averages within classes
%             s_group = s;
%             s_group(av_dim) = randix{cc}(aa);
%             
%             % Extract samples for current group
%             Xtmp = X(s_group{:});
%             
%             % Store result in X
%             s_new = s;
%             s_new(av_dim) = {pos};
%             tmp(s_new{:}) = squeeze(mean(Xtmp, av_dim));
%             pos = pos + 1;
%         end
%     end
%     
%     % Store averaged version back in X
%     X = tmp;
% end
