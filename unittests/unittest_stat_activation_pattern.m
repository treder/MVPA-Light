% unit test of mv_stat_activation_pattern

rng(42)
tol = 10e-10;

% Get Gaussian data
nsamples = 200;
nfeatures = 90;
nclasses = 2;
[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, [], [], 0);

% Covariance matrix
Sw= nsamples/2 * cov(X(clabel==1,:),1) + nsamples/2 * cov(X(clabel==2,:),1);

%% if PCA is used to get the w, the pattern should be equal to the PC
[V,D] = eig(Sw);

cf=[];
cf.w = V(:,1);

p = mv_stat_activation_pattern(cf, X, clabel);

print_unittest_result('PCA: w and pattern collinear', 1, corr(cf.w, p), tol);
