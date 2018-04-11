function [perf,result] = mv_searchlight(cfg, X, clabel)
% Classification using a feature searchlight approach can highlight which
% feature(s) are informative. To this end, classification is performed on 
% each feature separately. However, neighbouring features can enter the 
% classification together when a matrix of size [features x features]
% specifying the neighbours is provided.
%
% Usage:
% [perf, ...] = mv_searchlight(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] data matrix or a 
%                  [samples x features x time] matrix. In the latter case,
%                  either all time points are used as features (if
%                  average=0) or they are averaged to one time (average=1)
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with parameters:
% .metric       - classifier performance metric, default 'acc'. See
%                 mv_classifier_performance. If set to [] or 'none', the 
%                 raw classifier output (labels or dvals depending on 
%                 cfg.cf_output) for each sample is returned. 
% .nb          - [features x features] matrix specifying which features
%                are neighbours of each other.
%                          - EITHER - 
%                a GRAPH consisting of 0's and 1's. A 1 in the 
%                (i,j)-th element signifies that feature i and feature j 
%                are neighbours, and a 0 means they are not neighbours
%                            - OR -
%                a DISTANCE MATRIX, where larger values mean larger distance.
%                If no matrix is provided, every feature is only neighbour
%                to itself and classification is performed for each feature 
%                separately.
% .size        - if a nb matrix is provided, size defines the 
%                size of the 'neighbourhood' of a feature.
%                if nb is a graph, it gives the number of steps taken 
%                     through the nb matrix to find neighbours:
%                     0: only the feature itself is considered (no neighbours)
%                     1: the feature and its immediate neighbours
%                     2: the feature, its neighbours, and its neighbours'
%                     neighbours
%                     3+: neighbours of neighbours of neighbours etc
%                     (default 1)
%                if nb is a distance matrix, size defines the number of
%                     neighbouring features that enter the classification
%                     0: only the feature itself is considered (no neighbours)
%                     1: the feature and its first closest neighbour 
%                        according to the distance matrix
%                     2+: the 2 closest neighbours etc.
% .average     - if 1 and X is [samples x features x time], the time
%                dimension is averaged ot a single feature (default 0). If
%                0, each time point is used as a separate feature
% .normalise    - normalises the data across samples, for each time point 
%                 and each feature separately, using 'zscore' or 'demean' 
%                 (default 'zscore'). Set to 'none' or [] to avoid normalisation.
% .feedback     - print feedback on the console (default 1)
%
% CROSS-VALIDATION parameters:
% .CV           - perform cross-validation, can be set to 'kfold',
%                 'leaveout', 'holdout', or 'none' (default 'kfold')
% .K            - number of folds in k-fold cross-validation (default 5)
% .P            - if CV is 'holdout', P is the fraction of test samples
%                 (default 0.1)
% .stratify     - if 1, the class proportions are approximately preserved
%                 in each fold (default 1)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds (default 1)
%
% Returns:
% perf          - [features x 1] vector of classifier performances

% (c) Matthias Treder 2017

X = double(X);

mv_set_default(cfg,'nb',[]);
mv_set_default(cfg,'size',1);
mv_set_default(cfg,'metric','auc');
mv_set_default(cfg,'average',0);
mv_set_default(cfg,'feedback',1);
mv_set_default(cfg,'classifier','lda');
mv_set_default(cfg,'param',[]);
mv_set_default(cfg,'metric','acc');

% Cross-validation settings
mv_set_default(cfg,'CV','kfold');
mv_set_default(cfg,'repeat',5);
mv_set_default(cfg,'K',5);
mv_set_default(cfg,'P',0.1);
mv_set_default(cfg,'stratify',1);

switch(cfg.CV)
    case 'leaveout', cfg.K = size(X,1);
    case 'holdout', cfg.K = 1;
end

if cfg.average && ~ismatrix(X)
    X = mean(X,3);
end

[N, nFeat, ~] = size(X);

[clabel, nclasses] = mv_check_clabel(clabel);

perf = cell(nFeat,1);
perf_std = cell(nFeat,1);

%% Find the neighbourhood of the requested size
if isempty(cfg.nb)
    if cfg.feedback, fprintf('No neighbour matrix provided, considering each feature individually\n'), end
    % Do not include neighbours: each feature is only neighbour to itself
    nb = eye(nFeat); 
elseif numel(cfg.nb)==1 && cfg.nb == 0
    nb = eye(nFeat); 
else
    %%% Decide whether nb is a graph or a distance matrix
    if all(ismember([0,1],unique(cfg.nb))) % graph 
        
        % Use a trick used in transition matrices for Markov chains: taking the
        % i-th power of the matrix yields information about the neighbours that
        % can be reached in i steps
        nb = double(double(cfg.nb)^cfg.size > 0);
    
    else % distance matrix -> change it into a graph 
        % The resulting graph is not necessarily symmetric since if the
        % closest neighbour of chan1 is chan2, the closest neighbour of
        % chan2 can still be a different channel. Therefore, the matrix nb
        % contains the information of closest neighbours in its rows. E.g.,
        % the row nb(i,:) gives the closest neighbours of the i-th channel
        nb = zeros(nFeat);  % initialise as empty matrix
        for nn=1:nFeat
            [~,soidx] = sort(cfg.nb(nn,:),'ascend');
            % put 1's in the row corresponding to the nearest
            % neighbours
            nb(nn,soidx(1:cfg.size+1)) = 1;
        end
    end
    
end

%% Prepare cfg struct for mv_crossvalidate
tmp_cfg = cfg;
tmp_cfg.feedback = 0;

% Store the current state of the random number generator
rng_state = rng;

%% Loop across features
for ff=1:nFeat

    % Identify neighbours: multiply a unit vector with 1 at the ff'th with
    % the nb matrix, this yields the neighbours of feature ff
    u = [zeros(1,ff-1), 1, zeros(1,nFeat-ff)];
    neighbours = find( u * nb > 0);
    
    if cfg.feedback
        if numel(neighbours)>1
            fprintf('Classifying using feature %d with neighbours %s\n', ff, mat2str(setdiff(neighbours,ff)))
        else
            fprintf('Classifying using feature %d with no neighbours\n', ff)
        end
    end
    
    % Extract desired features and reshape into [samples x features]
    Xfeat = reshape(X(:,neighbours,:), N, []);

    % We always set the random number generator back to the same state:
    % this assures that the cross-validation folds are created in the same
    % way for each channel, increasing comparability
    rng(rng_state);
    
    % Perform cross-validation for specific feature(s)
    [perf{ff}, res] = mv_crossvalidate(tmp_cfg, Xfeat, clabel);
    perf_std{ff} = res.perf_std;
    
end

perf = cat(2,perf{:})';
perf_std = cat(2,perf_std{:})';

result = [];
if nargout>1
   result.function  = mfilename;
   result.perf      = perf;
   result.perf_std  = perf_std;
   result.metric    = cfg.metric;
   result.CV        = cfg.CV;
   result.K         = cfg.K;
   result.N         = size(X,1);
   result.repeat    = cfg.repeat;
   result.nclasses  = nclasses;
   result.classifier = cfg.classifier;
end