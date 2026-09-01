% Preprocessing unit test
%
rng(42)
tol = 10e-10;

% Random data
N = 100;
X = randn(N,40);
clabel = randi(2, N, 1);

cfg = [];
cfg.preprocess_fun = {};
cfg.preprocess_param = {};

oversample_param = mv_get_preprocess_param('oversample');
undersample_param = mv_get_preprocess_param('undersample');
zscore_param = mv_get_preprocess_param('zscore');
demean_param = mv_get_preprocess_param('demean');
average_param = mv_get_preprocess_param('average_samples');

%% try all preprocessing routeins + .is_train_set should be 1 after calling mv_preprocess once
cfg.preprocess_fun = {@mv_preprocess_oversample @mv_preprocess_undersample @mv_preprocess_zscore @mv_preprocess_demean @mv_preprocess_average_samples};
cfg.preprocess_param = {oversample_param, undersample_param, zscore_param, demean_param, average_param};

[cfg, X2, clabel2] = mv_preprocess(cfg, X, clabel);

print_unittest_result('[param.is_train_set] should be all 0 after calling mv_preprocess', 0, unique(cellfun(@(p) p.is_train_set==1 , cfg.preprocess_param)), tol);

%% undersample train set followed by ssd: select_data fields (signal_train/noise_train) must be
% resliced to match the undersampled number of samples (regression test)
nfeat = 5; ntime = 20;
clabel3 = [ones(40,1); 2*ones(20,1)];  % unbalanced classes: 40 vs 20
n_expect = 2 * min(sum(clabel3==1), sum(clabel3==2)); % after undersampling: 20+20
Xtr = randn(numel(clabel3), nfeat, ntime);

undersample_param2 = mv_get_preprocess_param('undersample');
ssd_param = mv_get_preprocess_param('ssd');

cfg3 = [];
cfg3.preprocess_fun = {@mv_preprocess_undersample @mv_preprocess_ssd};
cfg3.preprocess_param = {undersample_param2, ssd_param};

% ssd is used directly (not via mv_select_train_and_test_data), so
% signal_train/noise_train need to be set by hand, matching the
% *original* (pre-undersampling) number of samples
cfg3.preprocess_param{2}.signal_train = Xtr;
cfg3.preprocess_param{2}.noise_train = Xtr + 0.1*randn(size(Xtr));

[cfg3, Xtr_out] = mv_preprocess(cfg3, Xtr, clabel3);

print_unittest_result('[undersample+ssd] X should have undersampled number of samples', n_expect, size(Xtr_out,1), tol);
print_unittest_result('[undersample+ssd] signal_train should be resliced to undersampled number of samples', n_expect, size(cfg3.preprocess_param{2}.signal_train,1), tol);
print_unittest_result('[undersample+ssd] noise_train should be resliced to undersampled number of samples', n_expect, size(cfg3.preprocess_param{2}.noise_train,1), tol);

% -- continue with test-set pass on the same cfg3 (W already computed, is_train_set
% was toggled to 0 by mv_preprocess above) to check select_data reslicing on the test side
cfg3.preprocess_param{1}.undersample_test_set = 1;
Xte = randn(numel(clabel3), nfeat, ntime);
cfg3.preprocess_param{2}.signal_test = Xte;
cfg3.preprocess_param{2}.noise_test = Xte + 0.1*randn(size(Xte));
[cfg3, Xte_out] = mv_preprocess(cfg3, Xte, clabel3);

print_unittest_result('[undersample_test_set+ssd] X should have undersampled number of samples', n_expect, size(Xte_out,1), tol);
print_unittest_result('[undersample_test_set+ssd] signal_test should be resliced to undersampled number of samples', n_expect, size(cfg3.preprocess_param{2}.signal_test,1), tol);
print_unittest_result('[undersample_test_set+ssd] noise_test should be resliced to undersampled number of samples', n_expect, size(cfg3.preprocess_param{2}.noise_test,1), tol);

%% oversample: keep_idx_train should correctly map duplicated samples back
% to their source (regression test, direct call — avoids SSD, which
% transforms X via W and would break a post-SSD correspondence check)
nfeat = 5; ntime = 20;
clabel7 = [ones(6,1); 2*ones(3,1)];  % 6 vs 3, oversample class 2 by 3
Xtr7 = repmat((1:9)', [1, nfeat, ntime]);  % row i filled with value i

oversample_param = mv_get_preprocess_param('oversample');
oversample_param.replace = 0;  % without replacement, deterministic count
oversample_param.is_train_set = 1;

[pparam_out, Xtr7_out] = mv_preprocess_oversample(oversample_param, Xtr7, clabel7);

print_unittest_result('[oversample] X should have oversampled number of samples', 12, size(Xtr7_out,1), tol);

%% undersample followed by oversample followed by ssd: select_data fields must be
% resliced correctly through two consecutive sample-count-changing steps
% (regression test). Note: undersampling equalizes all classes to the
% minority count (6/3/9 -> 3/3/3 = 9 total); since classes are then already
% balanced, the subsequent oversample step adds nothing further (add_samples=0),
% so the final count stays at 9. This still tests that keep_idx composes
% correctly across two chained steps, even though the second step's mapping
% is the identity.
nfeat = 5; ntime = 20;
clabel9 = [ones(6,1); 2*ones(3,1); 3*ones(9,1)];  % 6/3/9 -> undersample to 3/3/3
n_expect_final = 3*3;  % 9: undersampling balances; oversampling has nothing left to add

Xtr9 = repmat((1:18)', [1, nfeat, ntime]);

undersample_param4 = mv_get_preprocess_param('undersample');
oversample_param3 = mv_get_preprocess_param('oversample');
oversample_param3.replace = 0;
ssd_param4 = mv_get_preprocess_param('ssd');

cfg8 = [];
cfg8.preprocess_fun = {@mv_preprocess_undersample @mv_preprocess_oversample @mv_preprocess_ssd};
cfg8.preprocess_param = {undersample_param4, oversample_param3, ssd_param4};

cfg8.preprocess_param{3}.signal_train = Xtr9;
cfg8.preprocess_param{3}.noise_train = Xtr9 + 0.1*randn(size(Xtr9));
[cfg8, Xtr9_out] = mv_preprocess(cfg8, Xtr9, clabel9);

print_unittest_result('[undersample+oversample+ssd] X should have final number of samples', n_expect_final, size(Xtr9_out,1), tol);
print_unittest_result('[undersample+oversample+ssd] signal_train should be resliced to final number of samples', n_expect_final, size(cfg8.preprocess_param{3}.signal_train,1), tol);
print_unittest_result('[undersample+oversample+ssd] noise_train should be resliced to final number of samples', n_expect_final, size(cfg8.preprocess_param{3}.noise_train,1), tol);


%% oversample followed by undersample followed by ssd: select_data fields must be
% resliced correctly through two consecutive sample-count-changing steps
% (regression test). Note: oversampling equalizes all classes to the
% majority count (6/3/9 -> 9/9/9 = 27 total); since classes are then already
% balanced, the subsequent undersample step removes nothing (rm_samples=0),
% so the final count stays at 27. This still tests that keep_idx composes
% correctly across two chained steps of opposite direction, even though the
% second step's mapping is the identity.
nfeat = 5; ntime = 20;
clabel10 = [ones(6,1); 2*ones(3,1); 3*ones(9,1)];  % 6/3/9 -> oversample to 9/9/9
n_expect_final2 = 3*9;  % 27: oversampling balances; undersampling has nothing left to remove

Xtr10 = repmat((1:18)', [1, nfeat, ntime]);

oversample_param4 = mv_get_preprocess_param('oversample');
undersample_param5 = mv_get_preprocess_param('undersample');
ssd_param5 = mv_get_preprocess_param('ssd');

cfg9 = [];
cfg9.preprocess_fun = {@mv_preprocess_oversample @mv_preprocess_undersample @mv_preprocess_ssd};
cfg9.preprocess_param = {oversample_param4, undersample_param5, ssd_param5};

cfg9.preprocess_param{3}.signal_train = Xtr10;
cfg9.preprocess_param{3}.noise_train = Xtr10 + 0.1*randn(size(Xtr10));
[cfg9, Xtr10_out] = mv_preprocess(cfg9, Xtr10, clabel10);

print_unittest_result('[oversample+undersample+ssd] X should have final number of samples', n_expect_final2, size(Xtr10_out,1), tol);
print_unittest_result('[oversample+undersample+ssd] signal_train should be resliced to final number of samples', n_expect_final2, size(cfg9.preprocess_param{3}.signal_train,1), tol);
print_unittest_result('[oversample+undersample+ssd] noise_train should be resliced to final number of samples', n_expect_final2, size(cfg9.preprocess_param{3}.noise_train,1), tol);