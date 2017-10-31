function [perf, result] = mv_classify_timextime(cfg, X, clabel, X2, clabel2)
% Time x time generalisation. A classifier is trained on the training data
% X and validated on either the same dataset X. Cross-validation is
% recommended to avoid overfitting. If another dataset X2 is provided,
% the classifier is trained on X and tested on X2. No cross-validation is
% performed in this case since the datasets are assumed to be independent.
%
% Usage:
% perf = mv_classify_timextime(cfg,X,clabel,<X2, clabel2>)
%
%Parameters:
% X              - [samples x features x time points] data matrix
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
% X2, clabel2    - (optional) second dataset with associated labels. If
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
%                 output (labels or dvals depending on cfg.cf_output) is returned.
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
% .feedback     - print feedback on the console (default 1)
%
% Returns:
% perf          - time1 x time2 classification matrix of classification
%                performance. 
% res           - struct with fields describing the classification result.
%                 Can be used as input to mv_statistics

% (c) Matthias Treder 2017

mv_set_default(cfg,'classifier','lda');
mv_set_default(cfg,'param',[]);
mv_set_default(cfg,'metric','acc');
mv_set_default(cfg,'CV','kfold');
mv_set_default(cfg,'repeat',5);
mv_set_default(cfg,'time1',1:size(X,3));
mv_set_default(cfg,'normalise','none');
mv_set_default(cfg,'feedback',1);

hasX2 = (nargin==5);
if hasX2, mv_set_default(cfg,'time2',1:size(X2,3));
else,     mv_set_default(cfg,'time2',1:size(X,3));
end

if isempty(cfg.metric) || any(ismember({'dval','auc','roc','tval'},cfg.metric))
    mv_set_default(cfg,'cf_output','dval');
else
    mv_set_default(cfg,'cf_output','clabel');
end

% Balance the data using oversampling or undersampling
mv_set_default(cfg,'balance','none');
mv_set_default(cfg,'replace',1);

if strcmp(cfg.CV,'kfold')
    mv_set_default(cfg,'K',5);
else
    mv_set_default(cfg,'K',1);
end

% Set non-specified classifier parameters to default
cfg.param = mv_get_classifier_param(cfg.classifier, cfg.param);

[~,~,clabel] = mv_check_labels(clabel);

nTime1 = numel(cfg.time1);
nTime2 = numel(cfg.time2);
nLabel = numel(clabel);

% Number of samples in the classes
N1 = sum(clabel == 1);
N2 = sum(clabel == 2);

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% Normalise
if strcmp(cfg.normalise,'zscore')
    X = zscore(X,[],1);
    if hasX2
        X2 = zscore(X2,[],1);
    end
elseif strcmp(cfg.normalise,'demean')
    X  = X  - repmat(mean(X,1), [size(X,1) 1 1]);
    if hasX2
        X2  = X2  - repmat(mean(X2,1), [size(X2,1) 1 1]);
    end
end

%% Time x time generalisation

% Save original data and labels in case we do over/undersampling
X_orig = X;
label_orig = clabel;

if ~strcmp(cfg.CV,'none') && ~hasX2
    % -------------------------------------------------------
    % One dataset X has been provided as input. X is hence used for both
    % training and testing. To avoid overfitting, cross-validation is
    % performed.
    if cfg.feedback, mv_print_classification_info(cfg,X,clabel); end

    % Initialise classifier outputs
    cf_output = cell(cfg.repeat, cfg.K, nTime1);
    testlabel = cell(cfg.repeat, cfg.K);
    
    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end

        % Undersample data if requested. We undersample the classes within the
        % loop since it involves chance (samples are randomly over-/under-
        % sampled) so randomly repeating the process reduces the variance
        % of the result
        if strcmp(cfg.balance,'undersample')
            [X,clabel] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);

        elseif isnumeric(cfg.balance)
            if ~all( cfg.balance <= [N1,N2])
                error(['cfg.balance is larger [%d] than the samples in one of the classes [%d, %d]. ' ...
                    'Concurrent over- and undersampling is currently not supported.'],cfg.balance,N1,N2)
            end
            % Sometimes we want to undersample to a specific
            % number (e.g. to match the number of samples across
            % subconditions)
            [X,clabel] = mv_balance_classes(X_orig,label_orig,cfg.balance,cfg.replace);
        end

        CV= cvpartition(clabel,cfg.CV,cfg.K);

        for kk=1:cfg.K                      % ---- CV folds ----
            if cfg.feedback, fprintf('%d ',kk), end

            % Train data
            Xtrain = X(CV.training(kk),:,:,:);

            % Get train and test labels
            trainlabel= clabel(CV.training(kk));
            testlabel{rr,kk} = clabel(CV.test(kk));

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
            Xtest= X(CV.test(kk),:,:);

            % permute and reshape into [ (trials x test times) x features]
            Xtest= permute(Xtest, [1 3 2]);
            Xtest= reshape(Xtest, CV.TestSize(kk)*nTime2, []);

            % ---- Training time ----
            for t1=1:nTime1

                % Training data for time point t1
                Xtrain_tt= squeeze(Xtrain(:,:,cfg.time1(t1)));

                % Train classifier
                cf= train_fun(cfg.param, Xtrain_tt, trainlabel);

                % Obtain classifier output (labels or dvals)
                cf_output{rr,kk,t1} = reshape( mv_get_classifier_output(cfg.cf_output, cf, test_fun, Xtest), sum(CV.test(kk)),[]);
            end

        end
        if cfg.feedback, fprintf('\n'), end
    end

    % Average classification performance across repeats and test folds
    avdim= [1,2];

elseif hasX2
    % -------------------------------------------------------
    % An additional dataset X2 has been provided. The classifier is trained
    % on X and tested on X2. No cross-validation is performed.
    if cfg.feedback
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
        cf= train_fun(cfg.param, Xtrain, clabel);

        % Obtain classifier output (labels or dvals)
        cf_output(:,t1,:) = reshape( mv_get_classifier_output(cfg.cf_output, cf, test_fun, Xtest), size(X2,1),[]);

    end

    testlabel = clabel2;
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
        cf= train_fun(cfg.param, Xtrain, clabel);

        for t2=1:nTime2      % ---- Testing time ----

            % Test data
            Xtest=  squeeze(X(:,:,cfg.time2(t2)));

            % Obtain the predicted class labels
            predlabels = test_fun(cf,Xtest);

            % Sum number of correctly predicted labels
            acc(t1,t2)= acc(t1,t2) + sum(predlabels(:) == clabel(:));
        end
    end

    acc = acc / nSam;

   % Calculate classifier performance
    if cfg.feedback, fprintf('Calculating classifier performance... '), end
    for mm=1:nMetrics
        perf{mm} = mv_classifier_performance(cfg.metric{mm}, cf_output, clabel);
    end
    avdim = [];

end

if isempty(cfg.metric)
    if cfg.feedback, fprintf('No performance metric requested, returning raw classifier output.\n'), end
    perf = cf_output;
    perf_std = [];
else
    if cfg.feedback, fprintf('Calculating classifier performance... '), end
    [perf, perf_std] = mv_calculate_performance(cfg.metric, cf_output, testlabel, avdim);
    if cfg.feedback, fprintf('finished\n'), end
end

result = [];
if nargout>1
   result.function  = mfilename;
   result.perf      = perf;
   result.perf_std  = perf_std;
   result.metric    = cfg.metric;
   result.CV        = cfg.CV;
   result.K         = cfg.K;
   result.repeat    = cfg.repeat;
   result.classifier = cfg.classifier;
end