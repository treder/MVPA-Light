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
X = zeros(nsamples, nfeatures, ntime);
[~,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

for tt=1:ntime
    X(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
end

% mv_classify_across_time
cfg = [];
cfg.feedback = 0;
acc1 = mv_classify_timextime(cfg, X, clabel);

% mv_classify
cfg.sample_dimension = 1;
cfg.feature_dimension = 2;
acc2 = mv_classify(cfg, X, clabel);

print_unittest_result('correlate with mv_classify_across_time', 1, corr(acc1,acc2), tol);

%% Create a dataset where classes can be perfectly discriminated for only some time points [4 classes]
nsamples = 100;
ntime = 100;
nfeatures = 10;
nclasses = 4;
prop = [];
scale = 0.0001;
do_plot = 0;

can_discriminate = 30:60;
cannot_discriminate = setdiff(1:ntime, can_discriminate);

% Generate data
X = zeros(nsamples, nfeatures, ntime);
[~,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

for tt=1:ntime
    if ismember(tt, can_discriminate)
        scale = 0.0001;
        if tt==can_discriminate(1)
            [X(:,:,tt),~,~,M]  = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
        else
            % reuse the class centroid to make sure that the performance
            % generalises
            X(:,:,tt)  = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot, M);
        end
    else
        scale = 10e1;
        X(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
    end
end

cfg = [];
cfg.feedback = 0;
cfg.classifier = 'multiclass_lda';

acc = mv_classify_timextime(cfg, X, clabel);

% performance should be around 100% for the discriminable time points, and
% around 25% for the non-discriminable ones
acc_discriminable = mean(mean( acc(can_discriminate, can_discriminate)));
acc_nondiscriminable = mean(mean( acc(cannot_discriminate, cannot_discriminate)));

tol = 0.03;
print_unittest_result('[4 classes] CV discriminable times = 1?', 1, acc_discriminable, tol);
print_unittest_result('[4 classes] CV non-discriminable times = 0.5?', 0.25, acc_nondiscriminable, tol);

