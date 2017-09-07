function varargout = mv_classify_across_time(cfg, X, label)
% Classification across time. A classifier is trained and validate for
% different time points in the dataset X. Cross-validation should be used
% to get a realistic estimate of classification performance.
%
% Usage:
% [perf, ...] = mv_classify_across_time(cfg,X,labels)
%
%Parameters:
% X              - [number of samples x number of features x number of time points]
%                  data matrix.
% labels         - [number of samples] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with hyperparameters:
% .classifier   - name of classifier, needs to have according train_ and test_
%                 functions (default 'lda')
% .param        - struct with parameters passed on to the classifier train
%                 function (default [])
% .metric       - classifier performance metric, default 'acc'. See
%                 mv_classifier_performance. If set to [], the raw classifier
%                 output (labels or dvals depending on cfg.output) for each
%                 sample is returned. Multiple metrics can be requested by
%                 providing a cell array e.g. {'acc' 'dval'}
% .CV           - perform cross-validation, can be set to
%                 'kfold' (recommended) or 'leaveout' (not recommended
%                 since it has a higher variance than k-fold) (default
%                 'kfold')
% .K            - number of folds (the K in K-fold cross-validation).
%                 For leave-one-out, K should be 1. (default 5 for kfold,
%                 1 for leave-one-out)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds. Only useful for CV = 'kfold'
%                 (default 1)
% .time         - indices of time points (by default all time
%                 points in X are used)
% .balance      - for imbalanced data with a minority and a majority class.
%                 'oversample' oversamples the minority class
%                 'undersample' undersamples the minority class
%                 such that both classes have the same number of samples
%                 (default 'none'). Note that for we undersample at the
%                 level of the repeats, whereas we oversample within each
%                 training set (for an explanation see mv_balance_classes).
%                 You can also give an integer number for undersampling.
%                 The samples will be reduced to this number. Note that
%                 concurrent over/undersampling (oversampling of the
%                 smaller class, undersampling of the larger class) is not
%                 supported at the moment
% .replace      - if balance is set to 'oversample' or 'undersample',
%                 replace deteremines whether data is drawn with
%                 replacement (default 1)
% .verbose      - print information on the console (default 1)
%
% Returns:
% perf          - [time x 1] vector of classifier performances. If multiple
%                 metrics were requested, multiple output arguments are
%                 provided.
%
% Note: For time x time generalisation, use mv_classify_timextime

% (c) Matthias Treder 2017

mv_setDefault(cfg,'classifier','lda');
mv_setDefault(cfg,'param',[]);
mv_setDefault(cfg,'metric','acc');
mv_setDefault(cfg,'CV','kfold');
mv_setDefault(cfg,'repeat',5);
mv_setDefault(cfg,'time',1:size(X,3));
mv_setDefault(cfg,'verbose',0);

if isempty(cfg.metric) || any(ismember({'dval','auc','roc'},cfg.metric))
    mv_setDefault(cfg,'output','dval');
else
    mv_setDefault(cfg,'output','label');
end

% Balance the data using oversampling or undersampling
mv_setDefault(cfg,'balance','none');
mv_setDefault(cfg,'replace',1);

if strcmp(cfg.CV,'kfold')
    mv_setDefault(cfg,'K',5);
else
    mv_setDefault(cfg,'K',1);
end

% Set non-specified classifier parameters to default
cfg.param = mv_classifier_defaults(cfg.classifier, cfg.param);

[~,~,label] = mv_check_labels(label);

nTime = numel(cfg.time);
nLabel = numel(label);

% Number of samples in the classes
N1 = sum(label == 1);
N2 = sum(label == 2);

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% Prepare performance metrics
if ~isempty(cfg.metric) && ~iscell(cfg.metric)
    cfg.metric = {cfg.metric};
end

nMetrics = numel(cfg.metric);
perf= cell(nMetrics,1);

%% Classify across time

% Save original data and labels in case we do over-/undersampling
X_orig = X;
label_orig = label;

if ~strcmp(cfg.CV,'none')
    if cfg.verbose, fprintf('Using %s cross-validation (K=%d) with %d repetitions.\n',cfg.CV,cfg.K,cfg.repeat), end

    % Initialise classifier outputs
    cf_output = cell(cfg.repeat, cfg.K, nTime);
    testlabel = cell(cfg.repeat, cfg.K);

    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.verbose, fprintf('Repetition #%d. Fold ',rr), end

        % Undersample data if requested. We undersample the classes within the
        % loop since it involves chance (samples are randomly over-/under-
        % sampled) so randomly repeating the process reduces the variance
        % of the result
        if strcmp(cfg.balance,'undersample')
            [X,label] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
        elseif isnumeric(cfg.balance)
            if ~all( cfg.balance <= [N1,N2])
                error(['cfg.balance is larger [%d] than the samples in one of the classes [%d, %d]. ' ...
                    'Concurrent over- and undersampling is currently not supported.'],cfg.balance,N1,N2)
            end
            % Sometimes we want to undersample to a specific
            % number (e.g. to match the number of samples across
            % subconditions)
            [X,label] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
        end

        CV= cvpartition(label,cfg.CV,cfg.K);

        for kk=1:cfg.K                      % ---- CV folds ----
            if cfg.verbose, fprintf('%d ',kk), end

            % Train data
            Xtrain = X(CV.training(kk),:,:,:);

            % Get training labels
            trainlabel= label(CV.training(kk));
            testlabel{rr,kk} = label(CV.test(kk));

            % Oversample data if requested. It is important to oversample
            % only the *training* set to prevent overfitting (see
            % mv_balance_classes for an explanation)
            if strcmp(cfg.balance,'oversample')
                [Xtrain,trainlabel] = mv_balance_classes(Xtrain,trainlabel,cfg.balance,cfg.replace);
            end

            for tt=1:nTime           % ---- Train and test time ----
                % Train and test data for time point tt
                Xtrain_tt= squeeze(Xtrain(:,:,cfg.time(tt)));
                Xtest= squeeze(X(CV.test(kk),:,cfg.time(tt)));

                % Train classifier
                cf= train_fun(cfg.param, Xtrain_tt, trainlabel);

                % Obtain classifier output (labels or dvals)
                cf_output{rr,kk,tt} = mv_classifier_output(cfg.output, cf, test_fun, Xtest);
                
            end
        end
        if cfg.verbose, fprintf('\n'), end
    end

    % Calculate classifier performance and average across all test sets
    % (i.e. all repeats and folds)
    if cfg.verbose, fprintf('Calculating classifier performance\n'), end
    for mm=1:nMetrics
        perf{mm} = mv_classifier_performance(cfg.metric{mm}, cf_output, label_orig, 2);
    end

else
    % No cross-validation, just train and test once for each
    % training/testing time. This gives the classification performance for
    % the training set, but it may lead to overfitting and thus to an
    % artifically inflated performance.

    % Initialise classifier outputs
    cf_output = nan(numel(label), nTime);

    % Rebalance data using under-/over-sampling if requested
    if ~strcmp(cfg.balance,'none')
        [X,label] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
    end

    for tt=1:nTime          % ---- Train and test time ----
        % Train and test data
        Xtraintest= squeeze(X(:,:,cfg.time(tt)));

        % Train classifier
        cf= train_fun(Xtraintest, label, cfg.param);
        
        % Obtain classifier output (labels or dvals)
        cf_output(:,tt) = mv_classifier_output(cfg.output, cf, test_fun, Xtraintest);
    end

    % Calculate classifier performance
    for mm=1:nMetrics
        perf{mm} = mv_classifier_performance(cfg.metric{mm}, cf_output, label);
    end
end

if nMetrics==0
    % If no metric was requested, return the raw classifier output
    varargout{1} = cf_output;
else
    varargout(1:nMetrics) = perf;
end

if cfg.verbose, fprintf('Finished\n'), end
