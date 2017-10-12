function perf = mv_searchlight(cfg, X, clabel)
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
% .metric      - classifier performance metric, default 'acc'. See
%                mv_classifier_performance. If set to [], the raw classifier
%                output (labels or dvals depending on cfg.output) for each
%                sample is returned
% .nb          - [features x features] matrix specifying the neighbours.
%                          - EITHER - 
%                a GRAPH consisting of 0's and 1's where a 1 in the 
%                (i,j)-th element signifies that feature i and feature j 
%                are neighbours.
%                            - OR -
%                a DISTANCE MATRIX, where each entry needs to have a value 
%                >= 0 and the larger values mean larger distance.
%                If no matrix is provided, every feature is only neighbour
%                to itself and classification is performed for each feature 
%                separately.
% .num         - if nb is a graph: num defines the number of steps taken 
%                     through the
%                     neighbourhood matrix to find neighbours:
%                     0: only the feature itself is considered (no neighbours)
%                     1: the feature and its immediate neighbours
%                     2: the feature, its neighbours, and its neighbours'
%                     neighbours
%                     3+: neighbours of neighbours of neighbours etc
%                     (default 1)
%                if nb is a distance matrix: num defines the number of
%                     neighbouring features that enter the classification
%                     0: only the feature itself is considered (no neighbours)
%                     1: the feature and its closest neighbour according to
%                     the distance matrix
%                     2+: the 2 closest neighbours etc.
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
mv_setDefault(cfg,'num',1);
mv_setDefault(cfg,'metric','auc');
mv_setDefault(cfg,'average',1);
mv_setDefault(cfg,'verbose',0);

if cfg.average && ~ismatrix(X)
    X = mean(X,3);
end

[N, nFeat, ~] = size(X);

perf = nan(nFeat,1);


%% Find the neighbours included by a stepsize of nbstep
if isempty(cfg.nb) || cfg.num == 0
    % Do not include neighbours: each feature is only neighbour to itself
    nb = eye(nFeat); 
else
    %%% Decide whether nb is a graph or a distance matrix
    if all(ismember([0,1],unique(cfg.nb))) % graph 
        
        % Use a trick used in transition matrices for Markov chains: taking the
        % i-th power of the matrix yields information about the neighbours that
        % can be reached in i steps
        nb = double(double(cfg.nb)^cfg.nbstep > 0);
    
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
            nb(nn,soidx(1:cfg.num+1)) = 1;
        end
    end
    
end

%% Prepare cfg struct for mv_crossvalidate
tmp_cfg = cfg;
tmp_cfg.verbose = 0;

% Store the current state of the random number generator
rng_state = rng;

%% Loop across features
for ff=1:nFeat

    % Identify neighbours: multiply a unit vector with 1 at the ff'th with
    % the nb matrix, this yields the neighbours of feature ff
    u = [zeros(1,ff-1), 1, zeros(1,nFeat-ff)];
    neighbourhood = find( u * nb > 0);
    
    if cfg.verbose, fprintf('Neighbours %s\n', mat2str(neighbourhood)), end
    
    % Extract desired features and reshape into [samples x features]
    Xfeat = reshape(X(:,neighbourhood,:), N, []);

    % We always set the random number generator back to the same state:
    % this assures that the cross-validation folds are created in the same
    % way for each channel, increasing comparability
    rng(rng_state);
    
    % Perform cross-validation
    perf(ff) = mv_crossvalidate(tmp_cfg, Xfeat, clabel);
    
end
