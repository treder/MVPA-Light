function perf = mv_searchlight(cfg, X, clabel)
% Classification using a feature searchlight approach can highlight which
% feature(s) are informative. To this end, classification is performed on 
% each feature separately. However, neighbouring features can enter the 
% classification together when a matrix of size [features x features]
% specifying the neighbours is provided.
% This matrix can either consist of 0's and 1's, with a 1 in the (i,j)-th
% element of the matrix meaning that feature i and feature j are
% neighbours, or it can be a distance matrix, specifying the distance
% between each pair of features. If no such matrix is provided,
% classification is performed on each feature separately.
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
% .metric      - classifier performance metric, default 'acc'. See
%                mv_classifier_performance. If set to [], the raw classifier
%                output (labels or dvals depending on cfg.output) for each
%                sample is returned
% .nb          - [features x features] matrix specifying the neighbours.
%                          - EITHER - 
%                consists of 0's and 1's where a 1 in the (i,j)-th element 
%                signifies that feature i and feature j are neighbours.
%                            - OR -
%                a distance matrix, where each entry needs to have a value 
%                >= 0 and the larger values mean larger distance.
%                If no matrix is provided, every feature is only neighbour
%                to itself.
% .nbstep      - defines the number of steps taken through the
%                neighbourhood matrix to find neighbours:
%                0: only the feature itself is considered (no neighbours)
%                1: the feature and its immediate neighbours
%                2: the feature, its neighbours, and its neighbours'
%                neighbours
%                3+: neighbours of neighbours of neighbours etc
%                (default 1)
% .max         - maximum number of neighbours considered for each classification
%                (default Inf). Can be used as an additional restriction
%                e.g. when step has a large value.
% .average     - if 1 and X is [samples x features x time], the time
%                dimension is averaged ot a single feature (default 1). If
%                0, each time point is used as a separate feature
%
% Additionally, you can pass on parameters for cross-validation. Refer to
% mv_crossvalidate for a description of cross-validation parameters.
%
% Returns:
% perf          - [features x 1] vector of classifier performance(s) 

% (c) Matthias Treder 2017

mv_setDefault(cfg,'nb',[]);
mv_setDefault(cfg,'nbstep',1);
mv_setDefault(cfg,'max',Inf);
mv_setDefault(cfg,'metric','auc');
mv_setDefault(cfg,'average',1);
mv_setDefault(cfg,'verbose',0);

if cfg.average && ~ismatrix(X)
    X = mean(X,3);
end

[N, nFeat, ~] = size(X);

perf = nan(nFeat,1);

%% Find the neighbours included by a stepsize of nbstep
if isempty(cfg.nb) || cfg.nbstep == 0
    % Do not include neighbours: each feature is only neighbour to itself
    cfg.nb = eye(nFeat); 
else
    % Use a trick used in transition matrices for Markov chains: taking the
    % i-th power of the matrix yields information about the neighbours that
    % can be reached in i steps
    cfg.nb = double(double(cfg.nb)^cfg.nbstep > 0);
end

%% Prepare cfg struct for mv_crossvalidate
tmp_cfg = cfg;
tmp_cfg.verbose = 0;

% Store the current state of the random number generator
rng_state = rng;

%% Loop across features
for ff=1:nFeat

    % Identify neighbours: multiply a unit vector with 1 at the ff'th with
    % the nb matrix, this yields the neighbourhood of feature ff
    u = [zeros(1,ff-1), 1, zeros(1,nFeat-ff)];
    neighbourhood = find( u * cfg.nb > 0);
    
    % If maximum number of neighbours is exceeded, we remove the excessive
    % neighbours
    if numel(neighbourhood) > cfg.max
        neighbourhood = neighbourhood(1:cfg.max);
    end
    
    % Extract desired features and reshape into [samples x features]
    Xfeat = reshape(X(:,neighbourhood,:), N, []);

    % We always set the random number generator back to the same state:
    % this assures that the cross-validation folds are created in the same
    % way for each channel, increasing comparability
    rng(rng_state);
    
    % Perform cross-validation
    perf(ff) = mv_crossvalidate(tmp_cfg, Xfeat, clabel);
    
end
