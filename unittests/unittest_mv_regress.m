% mv_regress unit test

rng(42)
tol = 10e-10;

%% Create a dataset where there is a linear relationship for the first half of the time points
nsamples = 100;
ntime = 300;
nfeatures = 10;
scale1 = 0.0001;  % low noise = strong relationship
scale2 = 1000;  % high noise = weak relationship

% Generate data with is noise free
X = zeros(nsamples, nfeatures, ntime);
[x, y] = simulate_regression_data('linear', nsamples, nfeatures, 0);

% add noise by hand to predictors
for tt=1:ntime
    if tt<ntime/2
        X(:,:,tt) = x + scale1 * randn(size(x));
    else
        X(:,:,tt) = randn(size(x)); % + scale2 * randn(size(x));
    end
end

cfg = [];
cfg.feedback = 0;
cfg.sample_dimension    = 1;
cfg.feature_dimension   = 2;
cfg.metric = 'mae';

% there should be a low MAE for the first half, and high MAE for the second
% half of the time points
mae = mv_regress(cfg, X, y);

% to test this, we correlate the MAE with a 0-1 step function
stepfun = zeros(ntime,1); 
stepfun(ntime/2+1:end) = 1;

print_unittest_result('correlation between 0-1 stepfunction and MAE > 0.95', corr(mae, stepfun), 1, 0.05);

% close all, plot(mae), hold all, plot(stepfun)

%% generalization [uncomment plotting to check]

% Generate data with is noise free
X = zeros(nsamples, nfeatures, ntime);
[x, y] = simulate_regression_data('linear', nsamples, nfeatures, 0);
% [x2, y2] = simulate_regression_data('linear', nsamples, nfeatures, 0);

% add noise by hand to predictors
for tt=1:ntime
    if tt<ntime/2
        X(:,:,tt) = x + scale1 * randn(size(x));
    else
        X(:,:,tt) =  randn(size(x));
    end
end

cfg.generalization_dimension = 3;

mae = mv_regress(cfg, X, y);

% close all, imagesc(mae), colorbar

%% Check different metrics and regression models [univariate]
N = 12;
X = randn(N, 5, 3, 4, 2);
Y = rand(N,1);

cfg = [];
cfg.sample_dimension     = 1;
cfg.feature_dimension    = 2;
cfg.generalization_dimension = 4;
cfg.metric = 'acc';

cfg.feedback = 0;

for metric = {'mae','mse','r_squared'}
    for model = {'ridge'}
        fprintf('[univariate] %s - %s\n', metric{:}, model{:})
        
        cfg.metric      = metric{:};
        cfg.model       = model{:};
        cfg.k           = 3;
        cfg.repeat      = 1;
        tmp = mv_regress(cfg, X, Y);
    end
end

%% Check different metrics and regression models [multivariate]
Y = rand(size(X,cfg.sample_dimension),5);

for metric = {'mae','mse','r_squared'}
    for model = {'ridge'}
        fprintf('[multivariate] %s - %s\n', metric{:}, model{:})
        
        cfg.metric      = metric{:};
        cfg.model       = model{:};
        cfg.k           = 3;
        cfg.repeat      = 1;
        tmp = mv_regress(cfg, X, Y);
    end
end

%% Check different cross-validation types [just run to check for errors]

% 4 input dimensions with 2 search dims
sz = [8, 26, 5, 3];
X = randn(sz);

cfg = [];
cfg.sample_dimension     = 2;
cfg.feature_dimension    = 3;
cfg.generalization_dimension    = 1;
cfg.cv                   = 'kfold';
cfg.k                    = 2;
cfg.repeat               = 2;
cfg.feedback             = 0;

Y = randn(sz(cfg.sample_dimension), 1); 

for cv = {'kfold' ,'leaveout', 'holdout', 'none'}
    fprintf('--%s--\n', cv{:})
    cfg.cv = cv{:};
    mv_regress(cfg, X, Y);
end

%% Check whether output dimensions are correct

% 4 input dimensions with 2 search dims
sz = [19, 2, 3, 40];
X = randn(sz);
Y = rand(sz(1), 1); 

cfg = [];
cfg.sample_dimension     = 1;
cfg.feature_dimension    = 3;
cfg.cv                   = 'kfold';
cfg.feedback             = 0;

perf = mv_regress(cfg, X, Y);
szp = size(perf);

print_unittest_result('is size(perf) correct for 4 input dimensions?', sz([2,4]), szp, tol);

% same but without cross-validation
cfg.cv                   = 'none';

perf = mv_regress(cfg, X, Y);
szp = size(perf);

print_unittest_result('[without crossval] is size(perf) correct for 4 input dimensions?', sz([2,4]), szp, tol);

%% Check whether output dimensions are correct cfg.flatten_features = 1 for 4D data

% 4 input dimensions with 2 search dims
sz = [9, 12, 2, 13];
X = randn(sz);
Y = randn(sz(1),1);

cfg = [];
cfg.sample_dimension    = 1;
cfg.feature_dimension   = [2 3];
cfg.cv                  = 'kfold';
cfg.k                   = 2;
cfg.feedback            = 0;
cfg.flatten_features    = 1;

perf = mv_regress(cfg, X, Y);
szp = size(perf);

print_unittest_result('size(perf) 4D data, 2 feature dim and cfg.flatten_features=1', [sz(4) 1], szp, tol);

%% Check whether output dimensions are correct cfg.flatten_features = 1 for 5D data

% 4 input dimensions with 2 search dims
sz = [9, 12, 2, 3, 4];
X = randn(sz);
Y = randn(sz(2),1);

cfg = [];
cfg.sample_dimension    = 2;
cfg.feature_dimension   = [1 5];
cfg.cv                  = 'kfold';
cfg.k                   = 2;
cfg.feedback            = 0;
cfg.flatten_features    = 1;

perf = mv_regress(cfg, X, Y);
szp = size(perf);

print_unittest_result('size(perf) 5D data, 2 feature dim and cfg.flatten_features=1', sz(3:4), szp, tol);

%% 5 input dimensions with 2 search dims + 1 generalization dim - are output dimensions as expected?
sz = [11, 8, 9, 7, 6];
X = randn(sz);

cfg = [];
cfg.model                   = 'ridge';
cfg.hyperparameter          = [];
cfg.hyperparameter.lambda   = 0.1;
cfg.feedback                = 0;
cfg.cv                      = 'kfold';
cfg.k                       = 2;
cfg.repeat                  = 1;

% try out all possible positions for samples, generalization, and features

nd = ndims(X);
for sd=1:nd   % sample dimension
    cfg.sample_dimension     = sd;
    Y = randn(size(X, sd), 1);

    for ff=1:nd-1  % feature dimension
        fd = mod(sd+ff-1,nd)+1;
        cfg.feature_dimension    = fd;
        search_dim = setdiff(1:nd, [sd, fd]);
    
        for gg=1:nd-2   % generalization dimension
            gd = search_dim(gg);
            cfg.generalization_dimension = gd;
            search_dim_without_gen = setdiff(search_dim, gd);
            perf = mv_regress(cfg, X, Y);
            szp = size(perf);
            print_unittest_result(sprintf('[5 dimensions] sample dim %d, feature dim %d, gen dim %d', sd, fd, gd), sz([search_dim_without_gen, gd ,gd]), szp, tol);
        end
    end
end


%% Check output size for searchlight with neighbour matrix that is non-square [1 search dim]
X = randn(15, 10, 19);
Y = randn(size(X,1),1);

% create random matrix with neighbours
nb1 = eye(size(X,2));
nb1 = nb1(1:end-4, :); % remove a few rows to make it non-square

cfg = [];
cfg.sample_dimension    = 1;
cfg.feature_dimension   = 3;
cfg.repeat              = 1;
cfg.feedback            = 0;
cfg.neighbours          = nb1;
perf = mv_regress(cfg, X, Y);

print_unittest_result('[1 search dim] size(perf) for non-square neighbours', size(nb1,1), size(perf,1), tol);

%% Check output size for searchlight with neighbour matrix that is non-square [2 search dim]
X = randn(19, 22, 19, 21);
Y = randn(size(X,1),1);

% create random matrix with neighbours
nb1 = eye(size(X,2));
nb2 = eye(size(X,4));

% remove a few rows
nb1 = nb1(1:end-8, :);
nb2 = nb2(1:end-1, :);

cfg = [];
cfg.sample_dimension    = 1;
cfg.feature_dimension   = 3;
cfg.repeat              = 1;
cfg.feedback            = 0;
cfg.neighbours          = {nb1 nb2};
perf = mv_regress(cfg, X, Y);

print_unittest_result('[2 search dim] size(perf) for non-square neighbours', [size(nb1,1) size(nb2,1)], size(perf), tol);


%% Transfer regression (cross regression)
nsamples2 = 60;
[x2, y2] = simulate_regression_data('linear', nsamples2, nfeatures, 0);

% Cross regression with the same dataset should yield the same result
% as cv = none
cfg = [];
cfg.cv = 'none';
cfg.feedback = 0;
perf1 = mv_regress(cfg, x, y);
perf2 = mv_regress(cfg, x, y, x, y);

print_unittest_result('transfer regression with same data vs cv=none', perf1, perf2, tol);

% add time dimension
x(:,:,80) = x;
x2(:,:,80) = x2;

cfg.generalization_dimension = 3;
perf1 = mv_regress(cfg, x, y);
perf2 = mv_regress(cfg, x, y, x, y);
print_unittest_result('transfer regression with same data vs cv=none (with generalization)', perf1, perf2, tol);

% sample dimension = 2 should not affect cross classification
x_perm = permute(x, [2 1 3]); % samples is now second dimension

perf      = mv_regress(cfg, x, y, x, y);
cfg.feature_dimension = 1;
cfg.sample_dimension = 2;
cfg.feedback = 0;
perf_perm = mv_regress(cfg, x_perm, y, x_perm, y);
print_unittest_result('transfer regression with sample dimension = 1 vs 2', perf, perf_perm, tol);

% cross decoding with generalization
x2(:,:,101) = squeeze(x2(:,:,1));

cfg = [];
cfg.generalization_dimension = 3;
cfg.feedback = 0;
perf   = mv_regress(cfg, x, y, x2, y2);

print_unittest_result('transfer regression with generalization', [size(x,3) size(x2,3)], size(perf), tol);



%% save: test for fields 'y_train' and 'model_param'
nsamples = 20;
nfeatures = 12;
ntime = 100;
X = zeros(nsamples, nfeatures, ntime);
[x, y] = simulate_regression_data('linear', nsamples, nfeatures, 0);

cfg = [];
cfg.repeat = 2;
cfg.k = 5;
cfg.feedback = 0;
cfg.save = {};
[~, result] = mv_regress(cfg, X, clabel);

print_unittest_result('[save={}] no y_train or model_param in result', true, (~isfield(result,'y_train'))&&(~isfield(result,'model_param')), tol);

cfg.save = {'y_train'};
[~, result] = mv_regress(cfg, X, clabel);
print_unittest_result('[save=y_train] y_train present', true, isfield(result,'y_train'), tol);
print_unittest_result('[save=y_train] model_param not present', false, isfield(result,'model_param'), tol);

cfg.save = {'y_train' 'model_param'};
[~, result] = mv_regress(cfg, X, clabel);
print_unittest_result('[save=y_train,model_param] y_train and model_param', true, isfield(result,'y_train')&&isfield(result,'model_param'), tol);

% add time dimension: now result.misc.model_param should have an extra dimension
X = randn(nsamples, nfeatures, ntime);
cfg.save = 'model_param';
[~, result] = mv_regress(cfg, X, clabel);
print_unittest_result('[save=model_param] misc.result.model_param for 3D data', [cfg.repeat, cfg.k, ntime], size(result.model_param), tol);

% no cross val
cfg.save = {'model_param' 'y_train'};
cfg.cv = 'none';
[~, result] = mv_regress(cfg, X, clabel);
print_unittest_result('[save=model_param, y_train, no crossval] misc.result.model_param', [1, 1, ntime], size(result.model_param), tol);

