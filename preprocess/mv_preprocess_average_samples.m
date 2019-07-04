function [preprocess_param, X, clabel] = mv_preprocess_average_samples(preprocess_param, X, clabel)
% Splits samples of the same class into groups and then replaces the data by
% the averages within each group. This increases SNR and reduces the sample 
% size. 
%
% For linear classification problems, this approach has been explored by
% Cichy and Pantazis (2017). In Treder (2018) it is generalized to non-linear 
% classification problems using kernels (called "kernel averaging") by 
% performing the averaging in Reproducing Kernel Hilbert Space (RKHS), in 
% which the data is again linearly separable. 
%
% To use kernel averaging, the input data must consist of kernel matrices
% instead of the samples (using a precomputed kernel). 
% sample_dimension has to indicate which of the dimensions code the samples. 
% For example, if the X consists of kernel matrices for each time point, ie 
% [samples x samples x time], then set sample_dimension = [1 2].
%
%Usage:
% [preprocess_param, X, clabel] = mv_preprocess_average_samples(preprocess_param, X, clabel)
%
%Parameters:
% X              - [... x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% preprocess_param - [struct] with preprocessing parameters
% .group_size       - group size, ie number of samples per group. If the
%                     total number of samples is not divisible by group_size, 
%                     some samples are discarded (default 5)
% .sample_dimension - which dimension(s) of the data matrix represent the samples
%                     (default 1)
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
sd = sort(preprocess_param.sample_dimension(:))';
sz = size(X);

% indices of samples per class
indices = arrayfun(@(c) find(clabel == c), 1:nclasses, 'Un', 0);

% number of samples per class
npc = cellfun(@numel, indices);

% number of averages per class
nav = floor(npc / preprocess_param.group_size);

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
        randix{cc}{aa} = tmpix((aa-1)*preprocess_param.group_size+1 : aa*preprocess_param.group_size);
    end
end

% there can be multiple sample dimensions. Therefore, we build a colon
% operator to extract the train/test samples irrespective of the
% position and number of sample dimensions
s = repmat({':'},[1, ndims(X)]);

% since there can be multiple sample dimensions, we average the data matrix
% dimension-by-dimension. In each iteration, one dimension is averaged and
% the result is being stored in tmp
for av_dim=sd  % -- loop across dimensions to be averaged
    
    % set size of dimension to number of averages
    sz(av_dim) = sum(nav);
    tmp = zeros(sz);
    pos = 1;
    
    % Average X
    for cc=1:nclasses     % -- loop across classes
        for aa=1:nav(cc)    % -- loop across averages within classes
            s_group = s;
            s_group(av_dim) = randix{cc}(aa);
            
            % Extract samples for current group
            Xtmp = X(s_group{:});
            
            % Store result in X
            s_new = s;
            s_new(av_dim) = {pos};
            tmp(s_new{:}) = squeeze(mean(Xtmp, av_dim));
            pos = pos + 1;
        end
    end
    
    % Store averaged version back in X
    X = tmp;
end

% Create class labels for averaged data
clabel = arrayfun(@(c) ones(nav(c),1)*c, 1:nclasses, 'Un',0);
clabel = cat(1, clabel{:});