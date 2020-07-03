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
    clabel = ones(sz(sd), 1); 
    clabel(ceil(end/2):end) = 2;

    for ff=1:nd-1  % feature dimension
        fd = mod(sd+ff-1,nd)+1;
        cfg.feature_dimension    = fd;
        search_dim = setdiff(1:nd, [sd, fd]);
    
        for gg=1:nd-2   % generalization dimension
            gd = search_dim(gg);
            cfg.generalization_dimension = gd;
            search_dim_without_gen = setdiff(search_dim, gd);
            perf = mv_classify(cfg, X, clabel);
            szp = size(perf);
            print_unittest_result(sprintf('[5 dimensions] sample dim %d, feature dim %d, gen dim %d', sd, fd, gd), sz([search_dim_without_gen, gd ,gd]), szp, tol);
        end
    end
end


