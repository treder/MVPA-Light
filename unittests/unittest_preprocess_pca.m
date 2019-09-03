% Preprocessing unit test
%
% pca
tol = 10e-10;

% Generate data
nsamples = 100;
nfeatures = 20;
nclasses = 2;
prop = [];
scale = 1;
do_plot = 0;

[X, clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

%% 2D example - check whether sizes are correct
X = randn(30,40);
n = 5;

% Get default parameters
pparam = mv_get_preprocess_param('pca');
pparam.target_dimension = 1;
pparam.feature_dimension = 2;
pparam.n = n;

[~, Xout] = mv_preprocess_pca(pparam, X);

print_unittest_result('size for 2D data with n=5', [size(X,1), n], size(Xout), tol);

pparam.target_dimension = 2;
pparam.feature_dimension = 1;
[~, Xout] = mv_preprocess_pca(pparam, X);

print_unittest_result('size for 2D data (transposed) with n=5', [n, size(X,2)], size(Xout), tol);

%% 5D example - check whether sizes are correct
sz = [11, 8, 9, 7, 6];
X = randn(sz);

pparam.n = 3;

% try out all possible positions for target and feature dimension
nd = ndims(X);
for sd=1:nd   % target dimension
    pparam.target_dimension = sd;

    for ff=1:nd-1  % feature dimension
        fd = mod(sd+ff-1,nd)+1;
        pparam.feature_dimension    = fd;
        
        [~, Xout] = mv_preprocess_pca(pparam, X);
        
        sz_expect = sz;
        sz_expect(fd) = pparam.n;
        
        print_unittest_result('size for 5D data with different target/feature dim', sz_expect, size(Xout), tol);
    end
end


%% check whether variance along target dimension is indeed 1 if normalize=1
sz = [25, 200];
X = randn(sz);

pparam.n = 6;
pparam.normalize = 1;
pparam.target_dimension = 1;
pparam.feature_dimension = 2;

[~, Xout] = mv_preprocess_pca(pparam, X);

print_unittest_result('[normalize=1] variances along target dim all 1', ones(1, pparam.n), var(Xout,[],1), tol);

%% check whether variances in decreasing order if normalize=0
sz = [25, 200];
X = randn(sz);

pparam.n = 4;
pparam.normalize = 0;
pparam.target_dimension = 2;
pparam.feature_dimension = 1;

[~, Xout] = mv_preprocess_pca(pparam, X);

print_unittest_result('[normalize=0] variances along target dim all decreasing', true(pparam.n-1, 1), diff(var(Xout,[],2))<0, tol);

%% covariance matrix = identity after PCA?
pparam = mv_get_preprocess_param('pca');
pparam.target_dimension = 1;
pparam.feature_dimension = 2;
pparam.n = 10;
pparam.normalize = 1;

[~, Xout] = mv_preprocess_pca(pparam, X);

print_unittest_result('PCA covariance = identity', eye(pparam.n), cov(Xout), tol);

%% nested preprocessing: train and test set [just run to see if there's errors]
nsamples = 200;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 1;
do_plot = 0;

[X, clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

Xtrain = X(1:100,:);            Xtest = X(101:end, :);
clabel_train = clabel(1:100);   clabel_test = clabel(101:end);


pparam = mv_get_preprocess_param('pca');
pparam.n = 4;
pparam.normalize = 1;
pparam.target_dimension = 1;
pparam.feature_dimension = 2;

% train set
[pparam, ~] = mv_preprocess_pca(pparam, Xtrain);

% apply on test set
pparam.is_train_set = 0;
mv_preprocess_pca(pparam, Xtest);

