% mv_classify unit test

rng(42)
tol = 10e-10;
mf = mfilename;

% In the first part we will replicate behaviour of the other high-level
% functions (mv_classify_across_time, mv_classify_timextime,
% mv_crossvalidate, mv_searchlight) and compare their output to mv_classify
% output
%% compare to mv_crossvalidate
nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 1;
do_plot = 0;
[X, clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

% mv_crossvalidate
rng(42)
cfg = [];
cfg.feedback    = 0;
acc1 = mv_crossvalidate(cfg, X, clabel);

% mv_crossvalidate
rng(42)
cfg.sample_dimension = 1;
cfg.feature_dimension = 2;
acc2 = mv_classify(cfg, X, clabel);

print_unittest_result('compare to mv_crossvalidate', acc1, acc2, tol);

%% compare to mv_classify_across_time
nsamples = 50;
ntime = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

% Generate data
X2 = zeros(nsamples, nfeatures, ntime);
[~,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

for tt=1:ntime
    X2(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
end

% mv_classify_across_time
rng(21)   % reset rng to get the same random folds
cfg = [];
cfg.feedback = 0;
acc1 = mv_classify_across_time(cfg, X2, clabel);

% mv_classify
rng(21)   % reset rng to get the same random folds
cfg.sample_dimension = 1;
cfg.feature_dimension = 2;
acc2 = mv_classify(cfg, X2, clabel);

print_unittest_result('compare to mv_classify_across_time', 0, norm(acc1-acc2), tol);

%% compare to mv_classify_timextime

% mv_classify_timextime
rng(22)
cfg = [];
cfg.feedback = 0;
acc1 = mv_classify_timextime(cfg, X2, clabel);

% mv_classify
rng(22)
cfg.sample_dimension = 1;
cfg.feature_dimension = 2;
cfg.generalization_dimension = 3;
acc2 = mv_classify(cfg, X2, clabel);

print_unittest_result('compare to mv_classify_timextime', 0, norm(acc1-acc2), tol);

%% compare to mv_searchlight

% mv_classify_timextime
rng(22)
cfg = [];
cfg.feedback = 0;
acc1 = mv_searchlight(cfg, X, clabel);

% mv_classify
rng(22)
cfg.sample_dimension = 1;
acc2 = mv_classify(cfg, X, clabel);

print_unittest_result('compare to mv_searchlight', 0, norm(acc1-acc2), tol);

%% Check different metrics and classifiers -- just run to see if there's errors (use 5 dimensions with 3 search dims)
X2 = randn(12, 5, 3, 4, 2);
clabel = ones(size(X2,1), 1); 
clabel(ceil(end/2):end) = 2;

cfg = [];
cfg.sample_dimension     = 1;
cfg.feature_dimension    = 2;
% % cfg.generalization_dimension = 4;
cfg.metric = 'acc';

cfg.feedback = 0;

for metric = {'acc','auc','f1','precision','recall','confusion','tval','dval'}
    for classifier = {'lda', 'logreg', 'multiclass_lda', 'svm', 'ensemble','kernel_fda'}
        if any(ismember(classifier,{'kernel_fda' 'multiclass_lda'})) && any(ismember(metric, {'tval','dval','auc'}))
            continue
        end
        fprintf('%s - %s\n', metric{:}, classifier{:})
        
        cfg.metric      = metric{:};
        cfg.classifier  = classifier{:};
        cfg.k           = 5;
        cfg.repeat      = 1;
        tmp = mv_classify(cfg, X2, clabel);
    end
end

%% Check whether output dimensions are correct

% 4 input dimensions with 2 search dims
sz = [19, 2, 3, 40];
X2 = randn(sz);
clabel = ones(sz(1), 1); 
clabel(ceil(end/2):end) = 2;

cfg = [];
cfg.sample_dimension     = 1;
cfg.feature_dimension    = 3;

perf = mv_classify(cfg, X2, clabel);
szp = size(perf);

print_unittest_result('is size(perf) correct for 4 input dimensions?', 0, norm(szp - sz([2,4])), tol);


% 5 input dimensions with 2 search dims + 1 generalization dim
sz = [20, 4, 10, 2, 3];
X2 = randn(sz);
clabel = ones(sz(1), 1); 
clabel(ceil(end/2):end) = 2;

cfg = [];
cfg.sample_dimension     = 5;
cfg.feature_dimension    = 1;
% cfg.generalization_dimension = 4;

perf = mv_classify(cfg, X2, clabel);
perf 


