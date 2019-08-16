% mv_classify_across_time unit test
%
rng(42)
tol = 10e-10;
mf = mfilename;

%% Create a dataset where classes can be perfectly discriminated for only some time points [two-class]
% 

nsamples = 100;
ntime = 300;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

can_discriminate = 50:100;

% Generate data
X = zeros(nsamples, nfeatures, ntime);
[~,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

for tt=1:ntime
    if ismember(tt, can_discriminate)
        scale = 0.0001;
        X(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
    else
        scale = 10e1;
        X(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
    end
end

cfg = [];
cfg.feedback = 0;

acc = mv_classify_across_time(cfg, X, clabel);

% plot(acc), title(mf,'interpreter','none')

% performance should be around 100% for the discriminable time points, and
% around 50% for the non-discriminable ones
acc_discriminable = mean( acc(can_discriminate));
acc_nondiscriminable = mean( acc(setdiff(1:ntime,can_discriminate)));

tol = 0.03;
print_unittest_result('[two-class] CV difference between (non-)/discriminable times', 0.5, acc_discriminable-acc_nondiscriminable, tol);


%% Create a dataset where classes can be perfectly discriminated for only some time points [4 classes]

nsamples = 100;
ntime = 300;
nfeatures = 10;
nclasses = 4;
prop = [];
scale = 0.0001;
do_plot = 0;

can_discriminate = 50:100;

% Generate data
X = zeros(nsamples, nfeatures, ntime);
[~,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

for tt=1:ntime
    if ismember(tt, can_discriminate)
        scale = 0.0001;
        X(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
    else
        scale = 10e1;
        X(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
    end
end

cfg = [];
cfg.feedback = 0;
cfg.classifier = 'multiclass_lda';

acc = mv_classify_across_time(cfg, X, clabel);

% performance should be around 100% for the discriminable time points, and
% around 25% for the non-discriminable ones
acc_discriminable = mean( acc(can_discriminate));
acc_nondiscriminable = mean( acc(setdiff(1:ntime,can_discriminate)));

tol = 0.03;
print_unittest_result('[4 classes] CV difference between (non-)/discriminable times', 0.75, acc_discriminable-acc_nondiscriminable, tol);

%% Check different metrics and classifiers for 2 classes -- just run to see if there's errors
nsamples = 60;
ntime = 2;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

% Generate data
[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
X(:,:,2) = X;

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
        tmp = mv_classify_across_time(cfg, X, clabel);
    end
end

%% same for 3 classes
nsamples = 60;
ntime = 2;
nfeatures = 10;
nclasses = 3;
prop = [];
scale = 0.0001;
do_plot = 0;

% Generate data
[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
X(:,:,2) = X + randn(size(X));
X(:,:,3:4) = X + randn(size(X));

cfg = [];
cfg.feedback = 0;

for metric = {'acc','confusion','f1','kappa','precision','recall'}
    for classifier = {'multiclass_lda','kernel_fda','naive_bayes'}
        fprintf('%s - %s\n', metric{:}, classifier{:})
        
        cfg.metric      = metric{:};
        cfg.classifier  = classifier{:};
        cfg.k           = 5;
        cfg.repeat      = 1;
        tmp = mv_classify_across_time(cfg, X, clabel);
    end
end

%% Check different cross-validation types [just run to check for errors]
sz = [30, 7, 100];
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
    mv_classify_across_time(cfg, X, clabel);
end