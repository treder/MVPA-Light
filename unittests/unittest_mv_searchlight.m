% mv_searchligh unit test

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
