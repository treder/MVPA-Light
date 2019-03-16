function [perf, result, testlabel] = mv_classify_across_time(cfg, X, clabel)
% Classification across time. A classifier is trained and validate for
% different time points in the dataset X. Cross-validation should be used
% to get a realistic estimate of classification performance.
%
% Usage:
% [perf, res] = mv_classify_across_time(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features x time points] data matrix
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with hyperparameters:
% .classifier   - name of classifier, needs to have according train_ and test_
%                 functions (default 'lda')
% .param        - struct with parameters passed on to the classifier train
%                 function (default [])
% .metric       - classifier performance metric, default 'accuracy'. See
%                 mv_classifier_performance. If set to [] or 'none', the 
%                 raw classifier output (labels, dvals or probabilities 
%                 depending on cfg.output_type) for each sample is returned. 
%                 Use cell array to specify multiple metrics (eg
%                 {'accuracy' 'auc'}
% .time         - indices of time points (by default all time
%                 points in X are used)
% .balance      - for imbalanced data that does not have
%                 the same number of instances in each class
%                 'oversample'          oversamples the minority classes
%                 'undersample'         undersamples the minority classes
%                 such that all classes have the same number of samples
%                 (default 'none'). Note that undersample occurs at the
%                 level of the repeats, whereas oversample occurs within each
%                 training set (for an explanation see mv_balance_classes).
%                 You can also give an integer number for undersampling.
%                 The samples will be reduced to this number. Note that
%                 concurrent over/undersampling (oversampling of the
%                 smaller class, undersampling of the larger class) is not
%                 supported at the moment
% .replace      - if balance is set to 'oversample' or 'undersample',
%                 replace determines whether data is drawn with
%                 replacement (default 1)
% .normalise    - normalises the data across samples, for each time point 
%                 and each feature separately, using 'zscore' or 'demean' 
%                 (default 'zscore'). Set to 'none' or [] to avoid normalisation.
% .feedback     - print feedback on the console (default 1)
%
%
% CROSS-VALIDATION parameters:
% .cv           - perform cross-validation, can be set to 'kfold',
%                 'leaveout', 'holdout', or 'none' (default 'kfold')
% .k            - number of folds in k-fold cross-validation (default 5)
% .p            - if cv is 'holdout', p is the fraction of test samples
%                 (default 0.1)
% .stratify     - if 1, the class proportions are approximately preserved
%                 in each fold (default 1)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds (default 1)
%
% Returns:
% perf          - [time x 1] vector of classifier performances. If
%                 metric='none', perf is a [r x k x t] cell array of
%                 classifier outputs, where each cell corresponds to a test
%                 set, k is the number of folds, r is the number of 
%                 repetitions, and t is the number of time points. If
%                 multiple metrics are requested, perf is a cell array
% result        - struct with fields describing the classification result.
%                 Can be used as input to mv_statistics and mv_plot_result
% testlabel     - [r x k] cell array of test labels. Can be useful if
%                 metric='none'
%
% Note: For time x time generalisation, use mv_classify_timextime

% (c) Matthias Treder

X = double(X);

mv_set_default(cfg,'classifier','lda');
mv_set_default(cfg,'param',[]);
mv_set_default(cfg,'metric','accuracy');
mv_set_default(cfg,'normalise','zscore');
mv_set_default(cfg,'time',1:size(X,3));
mv_set_default(cfg,'feedback',1);

% Cross-validation settings
mv_set_default(cfg,'cv','kfold');
mv_set_default(cfg,'repeat',5);
mv_set_default(cfg,'k',5);
mv_set_default(cfg,'p',0.1);
mv_set_default(cfg,'stratify',1);

switch(cfg.cv)
    case 'leaveout', cfg.k = size(X,1);
    case 'holdout', cfg.k = 1;
end

if any(ismember({'dval','auc','roc','tval'}, cfg.metric))
    mv_set_default(cfg,'output_type','dval');
else
    mv_set_default(cfg,'output_type','clabel');
end

if ~iscell(cfg.metric)
    cfg.metric = {cfg.metric};
end
nmetrics = numel(cfg.metric);

% Balance the data using oversampling or undersampling
mv_set_default(cfg,'balance','none');
mv_set_default(cfg,'replace',1);

% Set non-specified classifier parameters to default
cfg.param = mv_get_classifier_param(cfg.classifier, cfg.param);

[cfg, clabel, nclasses] = mv_check_inputs(cfg, X, clabel);

ntime = numel(cfg.time);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% Normalise
X = mv_normalise(cfg.normalise, X);

%% Classify across time

% Save original data and class labels in case we do over-/undersampling
X_orig = X;
label_orig = clabel;

if ~strcmp(cfg.cv,'none')
    if cfg.feedback, mv_print_classification_info(cfg,X,clabel); end

    % Initialise classifier outputs
    cf_output = cell(cfg.repeat, cfg.k, ntime);
    testlabel = cell(cfg.repeat, cfg.k);

    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end

        % Undersample data if requested. We undersample the classes within the
        % loop since it involves chance (samples are randomly over-/under-
        % sampled) so randomly repeating the process reduces the variance
        % of the result
        if strcmp(cfg.balance,'undersample')
            [X,clabel] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
        elseif isnumeric(cfg.balance)
            if numel(unique(sign(n - cfg.balance)))==2
                error(['cfg.balance [%d] is in between the sample sizes in the classes %s. ' ...
                    'Concurrent over- and undersampling is currently not supported.'],cfg.balance,mat2str(n))
            end
            % Sometimes we want to undersample to a specific
            % number (e.g. to match the number of samples across
            % subconditions)
            [X,clabel] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
        end

        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p);

        for kk=1:CV.NumTestSets                     % ---- CV folds ----
            if cfg.feedback, fprintf('%d ',kk), end

            % Train data
            Xtrain = X(CV.training(kk),:,:,:);

            % Get train and test class labels
            trainlabel= clabel(CV.training(kk));
            testlabel{rr,kk} = clabel(CV.test(kk));

            % Oversample data if requested. It is important to oversample
            % only the *training* set to prevent overfitting (see
            % mv_balance_classes for an explanation)
            if strcmp(cfg.balance,'oversample')
                [Xtrain,trainlabel] = mv_balance_classes(Xtrain,trainlabel,cfg.balance,cfg.replace);
            end

            for tt=1:ntime           % ---- Train and test time ----
                % Train and test data for time point tt
                Xtrain_tt= squeeze(Xtrain(:,:,cfg.time(tt)));
                Xtest= squeeze(X(CV.test(kk),:,cfg.time(tt)));

                % Train classifier
                cf= train_fun(cfg.param, Xtrain_tt, trainlabel);

                % Obtain classifier output (class labels, dvals or probabilities)
                cf_output{rr,kk,tt} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest);
                
            end
        end
        if cfg.feedback, fprintf('\n'), end
    end

    % Average classification performance across repeats and test folds
    avdim = [1,2];

else
    % No cross-validation, just train and test once for each
    % training/testing time. This gives the classification performance for
    % the training set, but it may lead to overfitting and thus to an
    % artifically inflated performance.
    
    if cfg.feedback
        fprintf('Training and testing on the same dataset (note: this can lead to overfitting).\n')
    end

    % Initialise classifier outputs
    cf_output = nan(numel(clabel), ntime);

    % Rebalance data using under-/over-sampling if requested
    if ~strcmp(cfg.balance,'none')
        [X,clabel] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
    end

    for tt=1:ntime          % ---- Train and test time ----
        % Train and test data
        Xtraintest= squeeze(X(:,:,cfg.time(tt)));

        % Train classifier
        cf= train_fun(cfg.param, Xtraintest, clabel);
        
        % Obtain classifier output (class labels or dvals)
        cf_output(:,tt) = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtraintest);
    end

    testlabel = clabel;
    avdim = [];
end

%% Calculate performance metrics
if cfg.feedback, fprintf('Calculating performance metrics... '), end
perf = cell(nmetrics, 1);
perf_std = cell(nmetrics, 1);
for mm=1:nmetrics
    if strcmp(cfg.metric{mm},'none')
        perf{mm} = cf_output;
        perf_std{mm} = [];
    else
        [perf{mm}, perf_std{mm}] = mv_calculate_performance(cfg.metric{mm}, cfg.output_type, cf_output, testlabel, avdim);
    end
end
if cfg.feedback, fprintf('finished\n'), end

if nmetrics==1
    perf = perf{1};
    perf_std = perf_std{1};
    cfg.metric = cfg.metric{1};
end

% if isempty(cfg.metric) || strcmp(cfg.metric{1},'none')
%     if cfg.feedback, fprintf('No performance metric requested, returning raw classifier output.\n'), end
%     perf = cf_output;
%     perf_std = [];
% else
%     if cfg.feedback, fprintf('Calculating classifier performance... '), end
%     [perf, perf_std] = mv_calculate_performance(cfg.metric, cfg.output_type, cf_output, testlabel, avdim);
%     if cfg.feedback, fprintf('finished\n'), end
% end

result = [];
if nargout>1
   result.function  = mfilename;
   result.perf      = perf;
   result.perf_std  = perf_std;
   result.metric    = cfg.metric;
   result.cv        = cfg.cv;
   result.k         = cfg.k;
   result.n         = size(X,1);
   result.repeat    = cfg.repeat;
   result.nclasses  = nclasses;
   result.classifier = cfg.classifier;
end