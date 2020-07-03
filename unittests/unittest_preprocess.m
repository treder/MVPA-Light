% Preprocessing unit test
%

rng(42)
tol = 10e-10;

% Random data
N = 100;
X = randn(N,40);
clabel = randi(2, N, 1);

cfg = [];
cfg.preprocess = {};
cfg.preprocess_param = {};

oversample_param = mv_get_preprocess_param('oversample');
undersample_param = mv_get_preprocess_param('undersample');
zscore_param = mv_get_preprocess_param('zscore');
demean_param = mv_get_preprocess_param('demean');
average_param = mv_get_preprocess_param('average_samples');

%% try all preprocessing routeins + .is_train_set should be 1 after calling mv_preprocess once
cfg.preprocess = {@mv_preprocess_oversample @mv_preprocess_undersample @mv_preprocess_zscore @mv_preprocess_demean @mv_preprocess_average_samples};
cfg.preprocess_param = {oversample_param, undersample_param, zscore_param, demean_param, average_param};

[cfg, X2, clabel2] = mv_preprocess(cfg, X, clabel);

print_unittest_result('[param.is_train_set] should be all 0 after calling mv_preprocess', 0, unique(cellfun(@(p) p.is_train_set==1 , cfg.preprocess_param)), tol);


