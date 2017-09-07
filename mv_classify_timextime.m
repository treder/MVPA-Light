function varargout = mv_classify_timextime(cfg, X, label, X2, label2)
% Time x time generalisation. A classifier is trained on the training data
% X and validated on either the same dataset X. Cross-validation is
% recommended to avoid overfitting. If another dataset X2 is provided,
% the classifier is trained on X and tested on X2. No cross-validation is
% performed in this case since the datasets are assumed to be independent.
%
% Usage:
% perf = mv_classify_timextime(cfg,X,label,<X2, label2>)
%
%Parameters:
% X              - [number of samples x number of features x number of time points]
%                  data matrix.
% label          - [number of samples] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
% X2, label2     - (optional) second dataset with associated labels. If
%                  provided, the classifier is trained on X and tested on
%                  X2 using
%
% cfg          - struct with optional parameters:
% .classifier   - name of classifier, needs to have according train_ and test_
%                 functions (default 'lda')
% .param        - struct with parameters passed on to the classifier train
%                 function (default [])
% .metric       - classifier performance metric, default 'acc'. See
%                 mv_classifier_performance. If set to [], the raw classifier
%                 output (labels or dvals depending on cfg.output) is returned.
%                 Multiple metrics can be requested by
%                 providing a cell array e.g. {'acc' 'dval'}
% .CV           - perform cross-validation, can be set to
%                 'kfold' (recommended) or 'leaveout' (not recommended
%                 since it has a higher variance than k-fold) (default
%                 'none')
% .K            - number of folds (the K in K-fold cross-validation).
%                 For leave-one-out, K should be 1. (default 5 for kfold,
%                 1 for leave-one-out)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds. Only useful for CV = 'kfold'
%                 (default 1)
% .time1        - indices of training time points (by default all time
%                 points in X are used)
% .time2        - indices of test time points (by default all time points
%                 in X are used)
% .balance      - for imbalanced data with a minority and a majority class.
%                 'oversample' oversamples the minority class
%                 'undersample' undersamples the minority class
%                 such that both classes have the same number of samples
%                 (default 'none'). Note that for we undersample at the
%                 level of the repeats, whereas we oversample within each
%                 training set (for an explanation see mv_balance_classes)
%                 You can also give an integer number for undersampling.
%                 The samples will be reduced to this number. Note that
%                 concurrent over/undersampling (oversampling of the
%                 smaller class, undersampling of the larger class) is not
%                 supported at the moment
% .replace      - if balance is set to 'oversample' or 'undersample',
%                 replace deteremines whether data is drawn with
%                 replacement (default 1)
% .normalise    - for evoked data is it recommended to normalise the samples
%                 across trials, for each time point and each sensor
%                 separately, using 'zscore' or 'demean' (default 'none')
% .verbose      - print information on the console (default 1)
%
% Returns:
% perf           - time1 x time2 classification matrix of classification
%                performance. If multiple metrics have been requested,
%                multiple output arguments are given

% (c) Matthias Treder 2017

mv_setDefault(cfg,'classifier','lda');
mv_setDefault(cfg,'param',[]);
mv_setDefault(cfg,'metric','acc');
mv_setDefault(cfg,'CV','kfold');
mv_setDefault(cfg,'repeat',5);
mv_setDefault(cfg,'time1',1:size(X,3));
mv_setDefault(cfg,'normalise','none');
mv_setDefault(cfg,'verbose',0);

hasX2 = (nargin==5);
if hasX2, mv_setDefault(cfg,'time2',1:size(X2,3));
else,     mv_setDefault(cfg,'time2',1:size(X,3));
end

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

nTime1 = numel(cfg.time1);
nTime2 = numel(cfg.time2);
nLabel = numel(label);

% Number of samples in the classes
N1 = sum(label == 1);
N2 = sum(label == 2);

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% Normalise
if strcmp(cfg.normalise,'zscore')
    X = zscore(X,[],1);
elseif strcmp(cfg.normalise,'demean')
    X  = X  - repmat(mean(X,1), [size(X,1) 1 1]);
end

%% Prepare performance metrics
if ~isempty(cfg.metric) && ~iscell(cfg.metric)
    cfg.metric = {cfg.metric};
end

nMetrics = numel(cfg.metric);

%% Time x time generalisation

% Save original data and labels in case we do over/undersampling
X_orig = X;
label_orig = label;

if ~strcmp(cfg.CV,'none') && ~hasX2
    % -------------------------------------------------------
    % One dataset X has been provided as input. X is hence used for both
    % training and testing. To avoid overfitting, cross-validation is
    % performed.
    if cfg.verbose, fprintf('Using %s cross-validation (K=%d) with %d repetitions.\n',cfg.CV,cfg.K,cfg.repeat), end

    % Initialise classifier outputs
    cf_output = cell(cfg.repeat, cfg.K, nTime1);
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

            % Oversample data if requested. We need to oversample each
            % training set separately to prevent overfitting (see
            % mv_balance_classes for an explanation)
            if strcmp(cfg.balance,'oversample')
                [Xtrain,trainlabel] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
            end

            % ---- Test data ----
            % Instead of looping through the second time dimension, we
            % reshape the data and apply the classifier to all time
            % points. We then need to apply the classifier only once
            % instead of nTime2 times.

            % Get test data
            Xtest= X(CV.test(ff),:,:);

            % permute and reshape into [ (trials x test times) x features]
            Xtest= permute(Xtest, [1 3 2]);
            Xtest= reshape(Xtest, CV.TestSize(ff)*nTime2, []);

            % ---- Training time ----
            for t1=1:nTime1

                % Training data for time point t1
                Xtrain_tt= squeeze(Xtrain(:,:,cfg.time1(t1)));

                % Train classifier
                cf= train_fun(cfg.param, Xtrain_tt, trainlabel);

                % Obtain classifier output (labels or dvals)
                cf_output{rr,kk,t1} = reshape( mv_classifier_output(cfg.output, cf, test_fun, Xtest), sum(CV.test(ff)),[]);

            end

        end
        if cfg.verbose, fprintf('\n'), end
    end

    testlabel = label_orig;
    avdim = 2;

elseif hasX2
    % -------------------------------------------------------
    % An additional dataset X2 has been provided. The classifier is trained
    % on X and tested on X2. No cross-validation is performed.
    if cfg.verbose
        fprintf('Training on X and testing on X2.\n')
        if ~strcmp(cfg.CV,'none'), fprintf('No cross-validation is performed, the cross-validation settings are ignored.\n'), end
    end

    % Initialise classifier outputs
    cf_output = nan(size(X2,1), nTime1, nTime2);

    % permute and reshape into [ (trials x test times) x features]
    Xtest= permute(X2, [1 3 2]);
    Xtest= reshape(Xtest, size(X2,1)*nTime2, []);

    % ---- Training time ----
    for t1=1:nTime1

        % Training data for time point t1
        Xtrain= squeeze(X(:,:,cfg.time1(t1)));

        % Train classifier
        cf= train_fun(cfg.param, Xtrain, label);

        % Obtain classifier output (labels or dvals)
        cf_output(:,t1,:) = reshape( mv_classifier_output(cfg.output, cf, test_fun, Xtest), size(X2,1),[]);

    end

    testlabel = label2;
    avdim = [];

else
    % -------------------------------------------------------
    % One dataset X has been provided as input. X is hence used for both
    % training and testing. However, cross-validation is not performed.
    % Note that this can lead to overfitting.

    error('Needs fixing: remove the second (t2) time loop and add performance metrics')

    for t1=1:nTime1          % ---- Training time ----
        % Training data
        Xtrain= squeeze(X(:,:,cfg.time1(t1)));

        % Train classifier
        cf= train_fun(cfg.param, Xtrain, label);

        for t2=1:nTime2      % ---- Testing time ----

            % Test data
            Xtest=  squeeze(X(:,:,cfg.time2(t2)));

            % Obtain the predicted class labels
            predlabels = test_fun(cf,Xtest);

            % Sum number of correctly predicted labels
            acc(t1,t2)= acc(t1,t2) + sum(predlabels(:) == label(:));
        end
    end

    acc = acc / nSam;

   % Calculate classifier performance
    if cfg.verbose, fprintf('Calculating classifier performance... '), end
    for mm=1:nMetrics
        perf{mm} = mv_classifier_performance(cfg.metric{mm}, cf_output, label);
    end
    avdim = [];

end

if nMetrics==0
    % If no metric was requested, return the raw classifier output
    varargout{1} = cf_output;
else
    % Calculate classifier performance, for each selected metric separately
    if cfg.verbose, fprintf('Calculating classifier performance... '), end
    varargout = cell(nMetrics,1);
    for mm=1:nMetrics
        varargout{mm} = mv_classifier_performance(cfg.metric{mm}, cf_output, testlabel, avdim);
    end
    if cfg.verbose, fprintf('finished\n'), end
end
