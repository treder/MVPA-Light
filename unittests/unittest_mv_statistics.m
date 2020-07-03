rng(42)
tol = 10e-10;

nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 1;
do_plot = 0;
[X, clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);


%% BINOMIAL: accuracy = 0.5 should give p=0.5 when N uneven
cfg = [];
cfg.test    = 'binomial';

% create fake result
result = [];
% provide multiple metrics: mv_statistics should choose the right one
result.metric = {'kappa' 'accuracy', 'dval'}; 
result.perf   = {-1, 0.5, 3};

n = [11, 101, 1001, 10001];

for nn=n
    result.n = nn;
    stat = mv_statistics(cfg, result);
    print_unittest_result(sprintf('[binomial] acc=0.5 should yield p=0.5 for N=%5.0f',nn), 0.5, stat.p, tol);
end

%% BINOMIAL: shape of stat.p should match shape of stat.perf
result.n = 100;
result.metric = 'acc';

result.perf = rand([5, 2]);
stat = mv_statistics(cfg, result);
print_unittest_result('[binomial] shape of stat.p matches result.perf', size(result.perf), size(stat.p), tol);

result.perf = rand([20, 5, 2]);
stat = mv_statistics(cfg, result);
print_unittest_result('[binomial] shape of stat.p matches result.perf', size(result.perf), size(stat.p), tol);

result.perf = rand([5,3,2,1,2,3]);
stat = mv_statistics(cfg, result);
print_unittest_result('[binomial] shape of stat.p matches result.perf', size(result.perf), size(stat.p), tol);

%% PERMUTATION: multiple metrics without cfg.metric set lead to error
cfg = [];
cfg.test    = 'permutation';

result.metric = {'kappa' 'accuracy', 'dval'}; 
result.perf   = {-1, 0.5, 3};

try 
    mv_statistics(cfg, result);
    err = false;
catch
    err = true;
end

print_unittest_result('[permutation] multiple metrics without cfg.metric set should lead to error', true, err, tol);

%% PERMUTATION: multiple metrics with cfg.metric should not cause an error
cfg = [];
cfg.test    = 'permutation';
cfg.metric  = 'kappa';
cfg.n_permutations = 0;

result.metric = {'kappa' 'accuracy', 'dval'}; 
result.perf   = {-1, 0.5, 3};
result.perf_dimension_names = {'x'};
result.function = 'mv_classify';

mv_statistics(cfg, result, X, clabel);

%% PERMUTATION: compare separable with non-separable time points
N = 1000;
P = 10;
T = 20;

X = randn(N, P, T);
clabel = [ones(N/2,1); 2*ones(N/2,1)];

% make first T/2 time points separable
X(1:N/2,:,1:T/2) = X(1:N/2,:,1:T/2) + 2;

% [X, clabel] = simulate_gaussian_data(N, P, 2, [], [], 0);
% X = repmat(X, [1 1 30]);
% [dat, clabel] = load_example_data('epoched1', 1);
% X = dat.trial;
cfg= [];
cfg.feedback = 0;
cfg.cv      = 'kfold';
cfg.k       = 10;
cfg.repeat  = 1;
[~, result] = mv_classify_across_time(cfg, X, clabel);

% permutation test should establish that indeed first T/2 times are
% separable, the others are not
cfg = [];
cfg.test            = 'permutation';
cfg.feedback        = 0;
cfg.n_permutations  = 50;
cfg.keep_null_distribution = 1;
stat = mv_statistics(cfg, result, X, clabel);

print_unittest_result('[permutation] separable vs non-separable time points', stat.mask, [ones(T/2,1); zeros(T/2,1)], 2); % allow for 2 mismatches
% [ stat.mask, [ones(T/2,1); zeros(T/2,1)] ]

%% CLUSTER PERMUTATION: check mask separable with non-separable time points
% time x time classification
cfg= [];
cfg.feedback = 0;
cfg.cv      = 'kfold';
cfg.k       = 10;
cfg.repeat  = 1;
[perf, result] = mv_classify_timextime(cfg, X, clabel);

cfg = [];
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.clustercritval  = 0.9;
cfg.feedback        = 0;
cfg.n_permutations  = 50;
stat = mv_statistics(cfg, result, X, clabel);

true_mask = zeros(size(X,3));
true_mask(1:size(X,3)/2, 1:size(X,3)/2) = 1;

print_unittest_result('[cluster permutation] separable vs non-separable', true_mask, double(stat.mask), 2); % allow for 2 mismatches

%% LEVEL 2 -- TODO --
