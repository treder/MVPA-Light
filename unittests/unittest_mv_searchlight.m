% mv_searchlight unit test

rng(42)
tol = 10e-10;

%% [2 classes] searchlight should identify the good features
% Generate data
nsamples = 200;
nfeatures = 40;

% Create Gaussian data with uncorrelated features
X = randn(nsamples, nfeatures);
clabel = zeros(nsamples, 1);
X(1:nsamples/2, :) = X(1:nsamples/2, :) + 5;
clabel(1:nsamples/2) = 1;
clabel(nsamples/2+1:end) = 2;

% Replace bad features by noise
bad_features = [11:20, 31:40];
good_features = setdiff(1:nfeatures, bad_features);
X(:, bad_features) = randn(nsamples, numel(bad_features));

cfg = [];
cfg.feedback    = 0;
acc = mv_searchlight(cfg, X, clabel);

% Bad features performance should be chance, good features about 100%

tol = 0.03;
print_unittest_result('[2 classes] Good features', 1, mean(acc(good_features)), tol);
print_unittest_result('[2 classes] Bad features', 0.5, mean(acc(bad_features)), tol);

%% adding neighbours should smear out classification performance - this should boost the bad features
nb = eye(nfeatures);
for ii=1:nfeatures
    sel_idx = max(1, ii-3):min(ii+3, nfeatures);
    nb(ii,sel_idx) = 1;
end

cfg.neighbours      = nb;
acc_nb = mv_searchlight(cfg, X, clabel);

print_unittest_result('Bad features with neighbours get better', 1, mean(acc(good_features))<mean(acc_nb(good_features)), tol);

%% compare different ways of specifying neighbours [distance matrix vs graph]

% specify as distance matrix
D = abs(randn(nfeatures));
D = D - diag(diag(D)); % set diagonal to 0
D = D + D';             % make symmetric

cfg = [];
cfg.neighbours      = D; % for distance matrix: the feature and its 2 closest neighbours
cfg.size            = 2;
cfg.feedback        = 0;
rng(12)
acc1 = mv_searchlight(cfg, X, clabel);

% specify as binary graph
G = zeros(size(D));
for ii=1:nfeatures
    % find cfg.size+1 smallest distances
    s = sort(D(ii,:));
    G(ii,:) = D(ii,:) <= s(cfg.size+1);
end

cfg.neighbours      = G;
cfg.size            = 1;  % for graph: feature and its neighbours
cfg.feedback        = 0;
rng(12)
acc2 = mv_searchlight(cfg, X, clabel);

print_unittest_result('results ', 0, norm(acc1-acc2), tol);

%% Check different metrics and classifiers for 2 classes -- just run to see if there's errors
cfg = [];
cfg.feedback = 0;

for metric = {'acc','auc','confusion','dval','f1','kappa','precision','recall','tval'}
    for classifier = {'lda', 'logreg', 'multiclass_lda', 'svm', 'ensemble','kernel_fda','naive_bayes'}
        if any(ismember(classifier,{'kernel_fda' 'multiclass_lda','naive_bayes'})) && any(ismember(metric, {'tval','dval','auc'}))
            continue
        end
        fprintf('%s - %s\n', metric{:}, classifier{:})
        
        cfg.metric      = metric{:};
        cfg.classifier  = classifier{:};
        cfg.k           = 5;
        cfg.repeat      = 1;
        tmp = mv_searchlight(cfg, X, clabel);
    end
end

%% same 3 classes
cfg = [];
cfg.feedback = 0;

for metric = {'acc','confusion','f1','kappa','precision','recall'}
    for classifier = {'multiclass_lda','kernel_fda','naive_bayes'}
        fprintf('%s - %s\n', metric{:}, classifier{:})
        
        cfg.metric      = metric{:};
        cfg.classifier  = classifier{:};
        cfg.k           = 5;
        cfg.repeat      = 1;
        tmp = mv_searchlight(cfg, X, clabel);
    end
end

%% Check different cross-validation types [just run to check for errors]
sz = [8, 26, 5];
X = randn(sz);

cfg = [];
cfg.cv                   = 'kfold';
cfg.k                    = 2;
cfg.p                    = 0.3;
cfg.feedback             = 0;

clabel = ones(sz(1), 1); 
clabel(ceil(end/2):end) = 2;

for cv = {'kfold' ,'leaveout', 'holdout', 'none'}
    fprintf('--%s--\n', cv{:})
    cfg.cv = cv{:};
    mv_searchlight(cfg, X, clabel);
end
