function varargout = mv_searchlight(cfg, X, clabel)
% Classification using a feature searchlight approach. To this end,
% classification is performed on each feature separately. Neighbouring
% features can enter the classification together when a matrix specifying
% the neighbours is provided.
%
% Usage:
% [perf, ...] = mv_searchlight(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] data matrix or a 
%                  [samples x features x time] matrix. In the latter case,
%                  all time points corresponding to a feature are 
%                  used as features.
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with parameters:
% .nb          - [features x features] matrix of 0's and 1's where a 1
%                in the (i,j)-th element signifies that feature i and
%                feature j are neighbours.
% .nbstep      - defines the number of steps taken through the
%                neighbourhood matrix to find neighbours:
%                0: only the feature itself is considered (no neighbours)
%                1: the feature and its immediate neighbours
%                2: the feature, its neighbours, and its neighbours'
%                neighbours
%                3+: neighbours of neighbours of neighbours etc
%                (default 1)
% .max         - maximum number of neighbours considered for each classification
%               (default Inf). Can be used as an additional restriction
%               e.g. when step has a large value.
%
% See mv_crossvalidate for cross-validation parameters.
%
% Returns:
% perf          - [features x 1] vector of classifier performance(s) 

% (c) Matthias Treder 2017

mv_setDefault(cfg,'nb',[]);
mv_setDefault(cfg,'nbstep',1);
mv_setDefault(cfg,'max',Inf);
mv_setDefault(cfg,'metric',{'auc'});
mv_setDefault(cfg,'verbose',0);

if ~isempty(cfg.metric) && ~iscell(cfg.metric), cfg.metric = {cfg.metric}; end
nMetric = numel(cfg.metric);

[N, nFeat, ~] = size(X);

%% Prepare metrics
perf = cell(nMetric,1);
tmp_perf  = cell(nMetric,1);  % holds the results for one feature iteration

% Initialise performances
for mm=1:nMetric
    perf{mm} = nan(nFeat,1);
end

%% Define neighbours included by a stepsize of nbstep
if isempty(cfg.nb) || cfg.nbstep == 0
    % Do not include neighbours: each feature is only neighbour to itself
    cfg.nb = eye(nFeat); 
else
    % Use a trick used in transition matrices for Markov chains: taking the
    % i-th power of the matrix yields information about the neighbours that
    % can be reached in i steps
    cfg.nb = double(cfg.nb^cfg.nbstep > 0);
end

%% Prepare cfg struct for mv_crossvalidate
tmp_cfg = cfg;
tmp_cfg.verbose = 0;

%% Loop across features
for ff=1:nFeat

    % Identify neighbours: multiply a unit vector with 1 at the ff'th with
    % the nb matrix, this yields the neighbourhood of feature ff
    u = [zeros(1,ff-1), 1, zeros(1,nFeat-ff)];
    neighbourhood = ( u * cfg.nb > 0);
    
    % Extract desired features and reshape into [samples x features]
    Xfeat = reshape(X(:,neighbourhood,:), N, []);
    
    % Perform cross-validation
    [tmp_perf{:}] = mv_crossvalidate(tmp_cfg, Xfeat, clabel);
    
    % Store results
    for mm=1:nMetric
        perf{mm}(ff) = tmp_perf{mm};
    end
end

varargout = perf;