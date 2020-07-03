% mv_crossvalidate unit test
rng(42)
tol = 10e-10;
mf = mfilename;

% Generate data
nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 1;
do_plot = 0;

[X, clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

%% accuracy on test set should be lower than on training set
cfg = [];
cfg.cv          = 'kfold';
cfg.feedback    = 0;

acc_kfold = mv_crossvalidate(cfg, X, clabel);

cfg.cv          = 'none';
acc_none = mv_crossvalidate(cfg, X, clabel);

tol = 0.03;
print_unittest_result('CV < no CV', 1, acc_kfold < acc_none, tol);

%% [2 classes] Cross-validation: performance for well-separated classes should be 100%
nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

cfg = [];
cfg.feedback        = 0;
cfg.metric          = 'acc';
cfg.classifier      = 'lda';
cfg.hyperparameter  = [];
cfg.hyperparameter.lambda    = 'auto';

actual = mv_crossvalidate(cfg, X, clabel);
expect = 1;

print_unittest_result('[2 classes] CV for well-separated data',expect, actual, tol);

%% [5 classes] Cross-validation: performance for well-separated classes should be 100%
nsamples = 120;
nfeatures = 10;
nclasses = 5;
prop = [];
scale = 0.0001;
do_plot = 0;

[X2,clabel2] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

cfg = [];
cfg.feedback        = 0;
cfg.metric          = 'acc';
cfg.classifier      = 'multiclass_lda';
cfg.hyperparameter           = [];
cfg.hyperparameter.lambda    = 'auto';

actual = mv_crossvalidate(cfg, X2, clabel2);
expect = 1;

print_unittest_result('[5 classes] CV for well-separated data',expect, actual, tol);

%% Check different metrics and classifiers for 2 classes -- just run to see if there's errors
cfg = [];
cfg.feedback = 0;

for metric = {'acc','auc','confusion','dval','f1','kappa','precision','recall','tval'}
    for classifier = {'lda', 'logreg', 'multiclass_lda', 'svm', 'ensemble','kernel_fda','naive_bayes'}
        if any(ismember(classifier,{'kernel_fda' 'multiclass_lda','naive_bayes'})) && any(ismember(metric, {'tval','dval','auc'}))
            continue
        end
        fprintf('%s - %s\n', metric{:}, classifier{:})
        
        cfg.metric      = metric{:};
        cfg.classifier  = classifier{:};
        cfg.k           = 5;
        cfg.repeat      = 1;
        tmp = mv_crossvalidate(cfg, X, clabel);
    end
end

%% same for 3 classes
cfg = [];
cfg.feedback = 0;

for metric = {'acc','confusion','f1','kappa','precision','recall'}
    for classifier = {'multiclass_lda','kernel_fda','naive_bayes'}
        fprintf('%s - %s\n', metric{:}, classifier{:})
        
        cfg.metric      = metric{:};
        cfg.classifier  = classifier{:};
        cfg.k           = 5;
        cfg.repeat      = 1;
        tmp = mv_crossvalidate(cfg, X, clabel);
    end
end

%% Check different cross-validation types [just run to check for errors]
sz = [30, 7];
X = randn(sz);

cfg = [];
cfg.cv                   = 'kfold';
cfg.k                    = 2;
cfg.p                    = 0.3;
cfg.feedback             = 0;

clabel = ones(sz(1), 1); 
clabel(ceil(end/2):end) = 2;

for cv = {'kfold' ,'leaveout', 'holdout', 'none'}
    fprintf('--%s--\n', cv{:})
    cfg.cv = cv{:};
    mv_crossvalidate(cfg, X, clabel);
end