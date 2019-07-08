function [pparam, X, clabel] = mv_preprocess_average_samples(pparam, X, clabel)
% Splits samples of the same class into groups and then replaces the data by
% the averages within each group. This increases SNR and reduces the sample 
% size. This approach has been explored by Cichy and Pantazis (2017).
%
% Note: this approach applies to linear classification problems. For
% a generalization to non-linear classification problems, see
% mv_preprocess_average_kernel.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_average_samples(pparam, X, clabel)
%
%Parameters:
% X              - [samples x ... x ...] data matrix
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

nclasses = max(clabel);
sz = size(X);

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

% SAMPLE AVERAGING

% size of averaged samples = number of averages
sz(1) = sum(nav);
tmp = zeros(sz);
pos = 1;

% Average X
for cc=1:nclasses     % -- loop across classes
    for aa=1:nav(cc)    % -- loop across averages within classes
        
        % Extract group and store average in tmp
        tmp(pos,:,:,:,:,:,:,:,:,:,:) = squeeze(mean(X(randix{cc}{aa},:,:,:,:,:,:,:,:,:,:), 1));
        pos = pos + 1;
    end
end

% Store averaged version back in X
X = tmp;


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
