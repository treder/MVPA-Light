% Classifier unit test
%
% Classifier: liblinear

% Note: the classifier itself is not being tested since it's external but
% rather the MVPA-Light interface. 

rng(42)   %% do not change - might affect the results
tol = 10e-10;
mf = mfilename;

%% first check whether LIBLINEAR is available
check = which('train','-all');
if isempty(check)
    warning('LIBLINEAR''s train() is not available or not in the path, skipping unit test')
    return
else
    try
        % this should work fine with liblinear but crash for Matlab's
        % train
        train(0,sparse(0),'-q');
    catch
        if numel(check)==1
            % there is an train but it seems to be Matlab's one
            warning('Found a train() function but it does not seem to be LIBLINEAR''s one, skipping unit test')
        else
            % there is multiple svmtrain functions
            warning('Found multiple functions called train: LIBLINEAR''s svmtrain() is either not available or overshadowed by another svmtrain function, skipping unit test')
        end
        return
    end
end

%%% Create Gaussian data
nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

[X_gauss,clabel_gauss] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

% Get classifier params
param = mv_get_hyperparameter('liblinear');

%% use cross-validation [no test, just look for crashes]
param = mv_get_hyperparameter('liblinear');
param.cv = 10;
cf = train_liblinear(param, X, clabel);

%% c = 1 to make a grid search
param.cv  = [];
param.c   = 1;
cf = train_liblinear(param, X, clabel);

%% set cost parameter by hand
param.c    = [];
param.cost = 1;
cf = train_liblinear(param, X, clabel);

predlabel = test_liblinear(cf, X);

% fprintf('Accuracy: %2.2f\n', sum(clabel==predlabel)/numel(clabel))

%% check classifier on multi-class spiral data: linear classifier should be near chance

% Create spiral data
N = 1000;
nrevolutions = 2;       % how often each class spins around the zero point
nclasses = 2;
prop = 'equal';
scale = 0;
[X_spiral, clabel_spiral] = simulate_spiral_data(N, nrevolutions, nclasses, prop, scale, 0);

%%% LINEAR kernel: cross-validation
cfg                 = [];
cfg.classifier      = 'liblinear';
cfg.hyperparameter  = [];
cfg.feedback        = 0;

acc_linear = mv_crossvalidate(cfg, X_spiral, clabel_spiral);

tol = 0.04;

% close to chance?
print_unittest_result('classif spiral data',1/nclasses, acc_linear, tol);
