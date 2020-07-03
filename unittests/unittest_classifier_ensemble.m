% Classifier unit test
%
% Classifier: ensemble

rng(42)
tol = 10e-10;

% Random data
X = randn(1000,100);
clabel = randi(2, size(X,1),1);

%% ensemble with 1 learner should be identical to testing the classifier directly [two-class LDA]
cfg = [];
cfg.feedback        = 0;
cfg.classifier      = 'ensemble';
cfg.hyperparameter = [];
cfg.hyperparameter.nfeatures = 0.9999;       % cannot set to 1 otherwise train_ensemble interprets it as a number rather than fraction
cfg.hyperparameter.nsamples  = 0.9999;
cfg.hyperparameter.nlearners = 1;
cfg.hyperparameter.strategy  = 'vote';

acc_ensemble = mv_crossvalidate(cfg, X, clabel);

cfg = [];
cfg.hyperparameter  = [];
cfg.feedback        = 0;
cfg.classifier      = 'lda';

acc_lda = mv_crossvalidate(cfg, X, clabel);

print_unittest_result('[two-class] 1-learner ensemble identical to lda directly', acc_ensemble, acc_lda, 0.02);

%% ensemble with 1 learner should be identical to testing the classifier directly [5 classes multi-class LDA]
nsamples = 1000;
nfeatures = 10;
nclasses = 5;
prop = [.2 .1 .25 .25 .2];
scale = 0.0001;
do_plot = 0;

[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);


cfg = [];
cfg.feedback        = 0;
cfg.classifier      = 'ensemble';
cfg.hyperparameter = [];
cfg.hyperparameter.learner   = 'multiclass_lda';
cfg.hyperparameter.nfeatures = 0.9999;       % cannot set to 1 otherwise train_ensemble interprets it as a number rather than fraction
cfg.hyperparameter.nsamples  = 0.9999;
cfg.hyperparameter.nlearners = 1;
cfg.hyperparameter.strategy  = 'vote';
cfg.hyperparameter.bootstrap = 0;

acc_ensemble = mv_crossvalidate(cfg, X, clabel);

cfg = [];
cfg.hyperparameter = [];
cfg.feedback        = 0;
cfg.classifier      = 'multiclass_lda';

acc_lda = mv_crossvalidate(cfg, X, clabel);

print_unittest_result('[multiclass] 1-learner ensemble identical to multiclass_lda directly', acc_ensemble, acc_lda, 0.02);

%% ensemble with 1 learner should give identical results whether strategy is 'dval' or 'vote' [two-class LDA]
nclasses = 2;
prop = [];
[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

cfg = [];
cfg.feedback        = 0;
cfg.classifier      = 'ensemble';
cfg.hyperparameter = [];
cfg.hyperparameter.nfeatures = 0.9999;
cfg.hyperparameter.nsamples  = 0.9999;
cfg.hyperparameter.nlearners = 1;

rng(42)
cfg.hyperparameter.strategy  = 'vote';
acc_vote = mv_crossvalidate(cfg, X, clabel);

rng(42)
cfg.hyperparameter.strategy  = 'dval';
acc_dval = mv_crossvalidate(cfg, X, clabel);

print_unittest_result('[two-class] 1 learner: same result for ''vote'' and ''dval''', acc_vote, acc_dval, tol);


