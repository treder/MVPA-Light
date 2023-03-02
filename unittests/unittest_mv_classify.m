% mv_classify unit test

rng(42)
tol = 10e-10;
mf = mfilename;

% In the first part we will replicate behaviour of the other high-level
% functions (mv_classify_across_time, mv_classify_timextime) to mv_classify
% output

nsamples = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 1;
do_plot = 0;
[X, clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

%% compare to mv_classify_across_time
nsamples = 50;
ntime = 100;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

% Generate data
X2 = zeros(nsamples, nfeatures, ntime);
[~,clabel2] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

for tt=1:ntime
    X2(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
end

% mv_classify_across_time
rng(21)   % reset rng to get the same random folds
cfg = [];
cfg.feedback = 0;
acc1 = mv_classify_across_time(cfg, X2, clabel2);

% mv_classify
rng(21)   % reset rng to get the same random folds
cfg.sample_dimension = 1;
cfg.feature_dimension = 2;
acc2 = mv_classify(cfg, X2, clabel2);

print_unittest_result('compare to mv_classify_across_time', acc1, acc2, tol);

%% compare to mv_classify_timextime

% mv_classify_timextime
rng(22)
cfg = [];
cfg.feedback = 0;
acc1 = mv_classify_timextime(cfg, X2, clabel2);

% mv_classify
rng(22)
cfg.sample_dimension = 1;
cfg.feature_dimension = 2;
cfg.generalization_dimension = 3;
acc2 = mv_classify(cfg, X2, clabel2);

print_unittest_result('compare to mv_classify_timextime', acc1, acc2, tol);

%% Check output size for searchlight with neighbour matrix that is non-square [1 search dim]
X = randn(30, 20, 29);
clabel = double(randn(size(X,1),1) > 0) + 1;

% create random matrix with neighbours
nb1 = eye(size(X,2));
nb1 = nb1(1:end-6, :); % remove a few rows to make it non-square

cfg = [];
cfg.sample_dimension    = 1;
cfg.feature_dimension   = 3;
cfg.repeat              = 1;
cfg.feedback            = 0;
cfg.neighbours          = nb1;
perf = mv_classify(cfg, X, clabel);

print_unittest_result('[1 search dim] size(perf) for non-square neighbours', size(nb1,1), size(perf,1), tol);

%% Check output size for searchlight with neighbour matrix that is non-square [2 search dim]
X = randn(30, 20, 15, 29);
clabel = double(randn(size(X,1),1) > 0) + 1;

% create random matrix with neighbours
nb1 = eye(size(X,2));
nb2 = eye(size(X,4));

% remove a few rows
nb1 = nb1(1:end-9, :);
nb2 = nb2(1:end-3, :);

cfg = [];
cfg.sample_dimension    = 1;
cfg.feature_dimension   = 3;
cfg.repeat              = 1;
cfg.feedback            = 0;
cfg.neighbours          = {nb1 nb2};
perf = mv_classify(cfg, X, clabel);

print_unittest_result('[2 search dim] size(perf) for non-square neighbours', [size(nb1,1) size(nb2,1)], size(perf), tol);


%% Create a dataset where classes can be perfectly discriminated for only some time points [two-class]
nsamples = 100;
ntime = 300;
nfeatures = 10;
nclasses = 2;
prop = [];
scale = 0.0001;
do_plot = 0;

can_discriminate = 50:100;
cannot_discriminate = setdiff(1:ntime, can_discriminate);

% Generate data
X = zeros(nsamples, nfeatures, ntime);
[~,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

for tt=1:ntime
    if ismember(tt, can_discriminate)
        scale = 0.0001;
        if tt==can_discriminate(1)
            [X(:,:,tt),~,~,M]  = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
        else
            % reuse the class centroid to make sure that the performance
            % generalises
            X(:,:,tt)  = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot, M);
        end
    else
        scale = 10e1; 
        X(:,:,tt) = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);
    end
end

cfg = [];
cfg.feedback = 0;
cfg.sample_dimension    = 1;
cfg.feature_dimension   = 2;
cfg.generalization_dimension   = 3;

[acc, result] = mv_classify(cfg, X, clabel);

% imagesc(acc), title(mf,'interpreter','none')
% figure,imagesc(acc)

% performance should be around 100% for the discriminable time points, and
% around 50% for the non-discriminable ones
acc_discriminable = mean(mean( acc(can_discriminate, can_discriminate)));
acc_nondiscriminable = mean(mean( acc(cannot_discriminate, cannot_discriminate)));

tol = 0.03;
print_unittest_result('[two-class] CV discriminable times = 1?', 1, acc_discriminable, tol);
print_unittest_result('[two-class] CV non-discriminable times = 0.5?', 0.5, acc_nondiscriminable, tol);


%% Check different metrics and classifiers for 2 classes -- just run to see if there's errors (use 5 dimensions with 3 search dims)
X2 = randn(12, 5, 3, 4, 2);
clabel = ones(size(X2,1), 1); 
clabel(ceil(end/2):end) = 2;

cfg = [];
cfg.sample_dimension     = 1;
cfg.feature_dimension    = 2;
cfg.generalization_dimension = 4;
cfg.metric = 'acc';

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
        tmp = mv_classify(cfg, X2, clabel);
    end
end

%% Check different metrics and classifiers for 3 classes -- just run to see if there's errors (use 5 dimensions with 3 search dims)
X2 = randn(18, 5, 5, 4, 2);
clabel = ones(size(X2,1), 1); 
clabel(7:end) = 2;
clabel(13:end) = 3;

cfg = [];
cfg.sample_dimension     = 1;
cfg.feature_dimension    = 2;
cfg.generalization_dimension = 4;
cfg.metric = 'acc';

cfg.feedback = 0;

for metric = {'acc','confusion','f1','kappa','precision','recall'}
    for classifier = {'multiclass_lda','kernel_fda','naive_bayes'}
        fprintf('%s - %s\n', metric{:}, classifier{:})
        
        cfg.metric      = metric{:};
        cfg.classifier  = classifier{:};
        cfg.k           = 5;
        cfg.repeat      = 1;
        tmp = mv_classify(cfg, X2, clabel);
    end
end

%% Check different cross-validation types [just run to check for errors]

% 4 input dimensions with 2 search dims
sz = [8, 26, 5, 3];
X2 = randn(sz);

cfg = [];
cfg.sample_dimension     = 2;
cfg.feature_dimension    = 3;
cfg.generalization_dimension    = 1;
cfg.cv                   = 'kfold';
cfg.k                    = 2;
cfg.feedback             = 0;

clabel = ones(sz(cfg.sample_dimension), 1); 
clabel(ceil(end/2):end) = 2;

for cv = {'kfold' ,'leaveout', 'holdout', 'none'}
    fprintf('--%s--\n', cv{:})
    cfg.cv = cv{:};
    mv_classify(cfg, X2, clabel);
end

%% Check whether output dimensions are correct for 4D data

% 4 input dimensions with 2 search dims
sz = [19, 2, 3, 40];
X2 = randn(sz);
clabel = ones(sz(1), 1); 
clabel(ceil(end/2):end) = 2;

cfg = [];
cfg.sample_dimension     = 1;
cfg.feature_dimension    = 3;
cfg.cv                   = 'kfold';
cfg.feedback             = 0;

perf = mv_classify(cfg, X2, clabel);
szp = size(perf);

print_unittest_result('size(perf) correct for 4D data', sz([2,4]), szp, tol);

% same but without cross-validation
cfg.cv                   = 'none';

perf = mv_classify(cfg, X2, clabel);
szp = size(perf);

print_unittest_result('[without crossval] is size(perf) correct for 4 input dimensions?', sz([2,4]), szp, tol);

%% Check whether output dimensions are correct cfg.flatten_features = 1 for 4D data

% 4 input dimensions with 2 search dims
sz = [9, 12, 2, 13];
X2 = randn(sz);
clabel = ones(sz(1), 1); 
clabel(ceil(end/2):end) = 2;

cfg = [];
cfg.sample_dimension    = 1;
cfg.feature_dimension   = [2 3];
cfg.cv                  = 'kfold';
cfg.k                   = 2;
cfg.feedback            = 0;
cfg.flatten_features    = 1;

perf = mv_classify(cfg, X2, clabel);
szp = size(perf);

print_unittest_result('size(perf) 4D data, 2 feature dim and cfg.flatten_features=1', [sz(4) 1], szp, tol);

%% Check whether output dimensions are correct cfg.flatten_features = 1 for 5D data

% 4 input dimensions with 2 search dims
sz = [9, 12, 2, 3, 4];
X2 = randn(sz);
clabel = ones(sz(2), 1); 
clabel(ceil(end/2):end) = 2;

cfg = [];
cfg.sample_dimension    = 2;
cfg.feature_dimension   = [1 5];
cfg.cv                  = 'kfold';
cfg.k                   = 2;
cfg.feedback            = 0;
cfg.flatten_features    = 1;

perf = mv_classify(cfg, X2, clabel);
szp = size(perf);

print_unittest_result('size(perf) 5D data, 2 feature dim and cfg.flatten_features=1', sz(3:4), szp, tol);

%% 5 input dimensions with 2 search dims + 1 generalization dim - are output dimensions as expected?
sz = [11, 8, 9, 7, 6];
X2 = randn(sz);

cfg = [];
cfg.classifier              = 'lda';
cfg.hyperparameter          = [];
cfg.hyperparameter.lambda   = 0.1;
cfg.feedback                = 0;
cfg.cv                      = 'kfold';
cfg.k                       = 2;
cfg.repeat                  = 1;

% try out all possible positions for samples, generalization, and features

nd = ndims(X2);
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
            perf = mv_classify(cfg, X2, clabel);
            szp = size(perf);
            print_unittest_result(sprintf('[5 dimensions] sample dim %d, feature dim %d, gen dim %d', sd, fd, gd), sz([search_dim_without_gen, gd ,gd]), szp, tol);
        end
    end
end


%% append dimensions - check dimensions 
nsamples = 40;
ntime = 20;
nfeatures = 5;
nclasses = 2;
prop = [];
scale = 0.0001;

% Generate data
X = zeros(nsamples, nfeatures, ntime, ntime+10);
[X ,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, 0);
X(:,:,ntime,ntime+10) = X;

% mv_classify_across_time
rng(21)   % reset rng to get the same random folds
cfg = [];
cfg.classifier              = 'naive_bayes';
cfg.sample_dimension        = 1;
cfg.feature_dimension       = 2;
cfg.feedback                = 0;
cfg.repeat                  = 1;
cfg.k                       = 2;

cfg.append                  = 1;
perf = mv_classify(cfg, X, clabel);

print_unittest_result('[cfg.append=1] size of perf', [size(X,3) size(X,4)], size(perf), tol);

%% append dimensions - check dimensions for cv = 'none'
rng(21)
cfg = [];
cfg.classifier              = 'naive_bayes';
cfg.sample_dimension        = 1;
cfg.feature_dimension       = 2;
cfg.feedback                = 0;
cfg.cv                      = 'none';

cfg.append                  = 1;
perf = mv_classify(cfg, X, clabel);

print_unittest_result('[cfg.append=1, cv=''none''] size of perf', [size(X,3) size(X,4)], size(perf), tol);

%% append vs no append should give same result
% appending dimensions should give the same result as not appending them,
% but it should be faster -> so far only possible with naive_bayes
nsamples = 20;
ntime = 20;
nfeatures = 5;
amplitudes = [5, 10];

X1 = simulate_erp_peak(nsamples, ntime, 10, 1, amplitudes(1), randn(nfeatures, 1)); % class 1
X2 = simulate_erp_peak(nsamples*2, ntime, 10, 1, amplitudes(2), randn(nfeatures, 1)); % class 2
X= [X1; X2];
clabel = [ones(nsamples, 1); ones(nsamples*2, 1)*2];

% mv_classify_across_time
cfg = [];
cfg.classifier              = 'naive_bayes';
cfg.sample_dimension        = 1;
cfg.feature_dimension       = 2;
cfg.feedback                = 0;
cfg.repeat                  = 1;
cfg.k                       = 2;

rng(21)   % reset rng to get the same random folds
cfg.append                  = 1;
perf1 = mv_classify(cfg, X, clabel);

rng(21)   % reset rng to get the same random folds
cfg.append                  = 0;
perf2 = mv_classify(cfg, X, clabel);

print_unittest_result('append vs no append should give same result', perf1, perf2, tol);

%% append vs no append should give same result, cv = 'none'
% mv_classify_across_time
rng(21)   % reset rng to get the same random folds
cfg = [];
cfg.classifier              = 'naive_bayes';
cfg.sample_dimension        = 1;
cfg.feature_dimension       = 2;
cfg.feedback                = 0;
cfg.cv                      = 'none';

cfg.append                  = 1;
perf1 = mv_classify(cfg, X, clabel);

cfg.append                  = 0;
perf2 = mv_classify(cfg, X, clabel);

print_unittest_result('[cv =''none''] append vs no append should give same result', perf1, perf2, tol);

%% Transfer classification (cross decoding)
tmp1 = simulate_erp_peak(nsamples, ntime, 10, 1, amplitudes(1), randn(nfeatures, 1)); % class 1
tmp2 = simulate_erp_peak(nsamples*2, ntime, 10, 1, amplitudes(2), randn(nfeatures, 1)); % class 2
X= [tmp1; tmp2];
clabel = [ones(nsamples, 1); ones(nsamples*2, 1)*2];

% Cross decoding with the same dataset should yield the same result
% as cv = none
cfg = [];
cfg.cv = 'none';
cfg.feedback = 0;
perf1 = mv_classify(cfg, X, clabel);
perf2 = mv_classify(cfg, X, clabel, X, clabel);

print_unittest_result('transfer classification with same data vs cv=none', perf1, perf2, tol);

cfg.generalization_dimension = 3;
perf1 = mv_classify(cfg, X, clabel);
perf2 = mv_classify(cfg, X, clabel, X, clabel);
print_unittest_result('transfer classification with same data vs cv=none (with generalization)', perf1, perf2, tol);

% sample dimension = 2 should not affect cross classification
X_perm = permute(X, [2 1 3]); % samples is now second dimension
X1 = X(1:2:end, :, :);
X2 = X(2:2:end, :, :);
X_perm1 = X_perm(:, 1:2:end, :);
X_perm2 = X_perm(:, 2:2:end, :);

perf      = mv_classify(cfg, X1, clabel(1:2:end), X2, clabel(2:2:end));
cfg.feature_dimension = 1;
cfg.sample_dimension = 2;
perf_perm = mv_classify(cfg, X_perm1, clabel(1:2:end), X_perm2, clabel(2:2:end));
print_unittest_result('transfer classification with sample dimension = 1 vs 2', perf, perf_perm, tol);

% dval: splitting the second dataset into separate classes should give the same result as testing on both classes at the same time
tmp1 = simulate_erp_peak(nsamples-10, ntime, 10, 1, amplitudes(1), randn(nfeatures, 1)); % class 1
tmp2 = simulate_erp_peak(nsamples*2-10, ntime, 10, 1, amplitudes(2), randn(nfeatures, 1)); % class 2
X2= [tmp1; tmp2];
clabel2 = [ones(nsamples-10, 1); ones(nsamples*2-10, 1)*2];

X1 = X(:,:,1:2);
X2 = X2(:,:,1:3);
clabel2_1 = clabel2(clabel2==1);
clabel2_2 = clabel2(clabel2==2);
X2_1 = X2(clabel2==1,:,:);
X2_2 = X2(clabel2==2,:,:);

cfg = [];
cfg.metric = 'dval';
cfg.generalization_dimension = 3;
cfg.feedback = 0;
perf   = mv_classify(cfg, X1, clabel, X2, clabel2);
perf_1 = mv_classify(cfg, X1, clabel, X2_1, clabel2_1);
perf_2 = mv_classify(cfg, X1, clabel, X2_2, clabel2_2);
print_unittest_result('transfer classification: testing both classes vs only class 1', perf(:,1), perf_1(:,1), tol);
print_unittest_result('transfer classification: testing both classes vs only class 2', perf(:,2), perf_2(:,2), tol);

% cross decoding with generalization
ntime2 = ntime + 10;
tmp1 = simulate_erp_peak(nsamples-10, ntime2, 10, 1, amplitudes(1), randn(nfeatures, 1)); % class 1
tmp2 = simulate_erp_peak(nsamples*2-10, ntime2, 10, 1, amplitudes(2), randn(nfeatures, 1)); % class 2
X2= [tmp1; tmp2];
clabel2 = [ones(nsamples-10, 1); ones(nsamples*2-10, 1)*2];

cfg = [];
cfg.generalization_dimension = 3;
cfg.feedback = 0;
perf   = mv_classify(cfg, X, clabel, X2, clabel2);

print_unittest_result('transfer classification with generalization', [size(X,3) size(X2,3)], size(perf), tol);

%% save: test for fields 'trainlabel' and 'model_param'
nsamples = 20;
nfeatures = 12;
ntime = 100;
X = randn(nsamples, nfeatures);
clabel = ones(nsamples,1);
clabel(1:2:end) = 2;

cfg = [];
cfg.repeat = 2;
cfg.k = 5;
cfg.feedback = 0;
cfg.save = {};
[~, result] = mv_classify(cfg, X, clabel);

print_unittest_result('[save={}] no trainlabel or model_param in result', true, (~isfield(result,'trainlabel'))&&(~isfield(result,'model_param')), tol);

cfg.save = {'trainlabel'};
[~, result] = mv_classify(cfg, X, clabel);
print_unittest_result('[save=trainlabel] trainlabel present', true, isfield(result,'trainlabel'), tol);
print_unittest_result('[save=trainlabel] model_param not present', false, isfield(result,'model_param'), tol);

cfg.save = {'trainlabel' 'model_param'};
[~, result] = mv_classify(cfg, X, clabel);
print_unittest_result('[save=trainlabel,model_param] trainlabel and model_param', true, isfield(result,'trainlabel')&&isfield(result,'model_param'), tol);

% add time dimension: now result.misc.model_param should have an extra dimension
X = randn(nsamples, nfeatures, ntime);
cfg.save = 'model_param';
[~, result] = mv_classify(cfg, X, clabel);
print_unittest_result('[save=model_param] misc.result.model_param for 3D data', [cfg.repeat, cfg.k, ntime], size(result.model_param), tol);

% no cross val
cfg.save = {'model_param' 'trainlabel'};
cfg.cv = 'none';
[~, result] = mv_classify(cfg, X, clabel);
print_unittest_result('[save=model_param, trainlabel, no crossval] misc.result.model_param', [1, 1, ntime], size(result.model_param), tol);
