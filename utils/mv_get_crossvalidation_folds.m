function CV = mv_get_crossvalidation_folds(cv, y, k, stratify, frac, fold, preprocess, preprocess_param)
% Defines a cross-validation scheme and returns a cvpartition object with
% the definition of the folds.
%
% Usage:
% CV = mv_get_crossvalidation_folds(cv, y, k, stratify, frac, fold)
%
%Parameters:
% cv          - cross-validation type:
%               'kfold':     K-fold cross-validation. The parameter K specifies
%                            the number of folds
%               'leaveout': leave-one-out cross-validation
%               'holdout':  Randomly splits data just once into training and
%                            hold-out/test set
%               'predefined': uses folds predefined by the user. In
%                            this case, fold needs to be set. You should
%                            set cfg.repeat = 1 since there is no
%                            randomness.
% y           - vector of class labels or regression outputs
% k           - number of folds (the k in k-fold) (default 5)
% stratify    - if 1, class proportions are roughly preserved in
%               each fold (default 0)
% frac        - if cv_type is 'holdout', frac is the fraction of test samples
%                 (default 0.1)
% fold        - if cv_type='predefined', fold is a vector of length
%                 #samples containing of 1's, 2's, 3's etc that specifies 
%                 for each sample the fold that it belongs to
% preprocess, preprocess_param  - preprocessing field. This allows cross-validation to be
%               aware of preprocessing operations such as average_samples.
%
%Output:
% CV - struct with cross-validation folds

% (c) Matthias Treder

N = size(y,1);

if nargin < 3,      k = 5; end
if nargin < 4,      stratify = 0; end
if nargin < 5,      frac = 0.1; end

switch(cv)
    case 'kfold'
        if stratify
            CV= cvpartition(y,'kfold', k);
        else
            CV= cvpartition(N, 'kfold', k);
        end
        
    case 'leaveout'
        averaging_idx = find(ismember(preprocess, {'average_samples', 'average_kernel'}));
        if any(averaging_idx)
            % We need to leave out one *averaged* sample, i.e. the test set
            % actually has to consist of multiple samples (equal to
            % averaging group_size) from the same class.
            % if samples are being averaged, we need to hold an *averaged*
            % sample out, not just a single sample
            group_size = preprocess_param{averaging_idx}.group_size;
            CV = struct();
            CV.group        = nan(length(y),1);
            nclasses = max(y);
            group_index = 1;
            for c=1:nclasses
                idx = find(y == c);  % indices of all samples of class c
                idx = idx(randperm(length(idx))); % shuffle indices
                for i=1:group_size:numel(idx)-group_size+1
                    CV.group(idx(i:i+group_size-1)) = group_index;
                    group_index = group_index + 1;
                end
            end
            CV.NumTestSets  = max(CV.group);
            CV.NumObservations  = length(y);
            CV.training     = @(x) ~ismember(CV.group, x);
            CV.test         = @(x) ismember(CV.group, x);
            CV.TrainSize    = arrayfun(@(x) sum(CV.training(x)), 1:CV.NumTestSets);
            CV.TestSize     = arrayfun(@(x) sum(CV.test(x)), 1:CV.NumTestSets);
        else
            CV= cvpartition(N,'leaveout');
        end

    case 'holdout'
        if stratify
            CV= cvpartition(y,'holdout',frac);
        else
            CV= cvpartition(N,'holdout',frac);
        end
        
    case 'predefined'
        CV = struct();
        u = unique(fold);
        n_groups = numel(u);
        CV.u            = u;
        CV.group        = fold;
        CV.NumTestSets  = numel(u);
        CV.NumObservations  = numel(fold);
        if iscell(fold)
            CV.training     = @(x) ~ismember(CV.group, CV.u(x));
            CV.test         = @(x) ismember(CV.group, CV.u(x));
            CV.TrainSize    = arrayfun(@(x) sum(~ismember(fold, u(x))), 1:n_groups);
            CV.TestSize     = arrayfun(@(x) sum(ismember(fold, u(x))), 1:n_groups);
        else
            CV.training     = @(x) CV.group ~= CV.u(x);
            CV.test         = @(x) CV.group == CV.u(x);
            CV.TrainSize    = arrayfun(@(x) sum(fold ~= u(x)), 1:n_groups);
            CV.TestSize     = arrayfun(@(x) sum(fold == u(x)), 1:n_groups);
        end
    otherwise error('Unknown cross-validation type: %s',cv)
end
