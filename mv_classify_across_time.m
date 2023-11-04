function [perf, result, testlabel] = mv_classify_across_time(cfg, X, clabel, varargin)
% Classification across time. A classifier is trained and tested for
% different time points in the dataset. Cross-validation can be used
% to get a realistic estimate of classification performance.
%
% Usage:
% [perf, res] = mv_classify_across_time(cfg, X, clabel, <X2, clabel2>)
%
%Parameters:
% X              - [samples x features x time points] data matrix -OR-
%                  [samples x samples  x time points] kernel matrices
% clabel         - [samples x 1] vector of class labels
% X2, clabel2    - (optional) if a second dataset is provided, transfer
%                  classification (aka cross decoding) is performed. 
%                  X/clabel acts as train data and X2/clabel2 acts as test 
%                  data. The datasets must have the same number of features and time points.
%
% cfg          - struct with optional parameters:
% .classifier   - name of classifier, needs to have according train_ and test_
%                 functions (default 'lda')
% .hyperparameter - struct with parameters passed on to the classifier train
%                 function (default [])
% .metric       - classifier performance metric, default 'accuracy'. See
%                 mv_classifier_performance. If set to [] or 'none', the 
%                 raw classifier output (labels, dvals or probabilities 
%                 depending on cfg.output_type) for each sample is returned. 
%                 Use cell array to specify multiple metrics (eg
%                 {'accuracy' 'auc'}
% .time         - indices of time points used e.g. 1:10 (by default all time
%                 points in X are used)
% .feedback     - print feedback on the console (default 1)
% .save         - use to save labels or model parameters for each train iteration.
%                 The cell array can contain  'trainlabel', 'model_param' 
%                 (ie classifier parameters) (default {}). 
%                 The results struct then contains the eponymous fields.
%
% CROSS-VALIDATION parameters:
% .cv           - perform cross-validation, can be set to 'kfold',
%                 'leaveout', 'holdout', 'predefined' or 'none' (default 'kfold')
% .k            - number of folds in k-fold cross-validation (default 5)
% .p            - if cv is 'holdout', p is the fraction of test samples
%                 (default 0.1)
% .stratify     - if 1, the class proportions are approximately preserved
%                 in each fold (default 1)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds (default 1)
% .fold         - if cv='predefined', fold is a vector of length
%                 #samples that specifies the fold each sample belongs to
%
% PREPROCESSING parameters:
% .preprocess         - cell array containing the preprocessing pipeline. The
%                       pipeline is applied in chronological order (default {})
% .preprocess_param   - cell array of preprocessing parameter structs for each
%                       function. Length of preprocess_param must match length
%                       of preprocess (default {})
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

% (c) Matthias Treder

X = double(X);
if ndims(X)~= 3, error('X must be 3-dimensional'), end

mv_set_default(cfg,'classifier','lda');
mv_set_default(cfg,'hyperparameter',[]);
mv_set_default(cfg,'metric','accuracy');
mv_set_default(cfg,'time',1:size(X,3));
mv_set_default(cfg,'feedback',1);
mv_set_default(cfg,'save',{});

mv_set_default(cfg,'preprocess',{});
mv_set_default(cfg,'preprocess_param',{});

[cfg, clabel, n_classes, n_metrics, clabel2] = mv_check_inputs(cfg, X, clabel, varargin{:});

has_second_dataset = (nargin==5);
if has_second_dataset
    X2 = double(varargin{1});
end
    
ntime = numel(cfg.time);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:n_classes);

% indicates whether the data represents kernel matrices
mv_set_default(cfg,'is_kernel_matrix', isfield(cfg.hyperparameter,'kernel') && strcmp(cfg.hyperparameter.kernel,'precomputed'));
if cfg.is_kernel_matrix,  mv_set_default(cfg,'dimension_names',{'samples','samples','time points'});
else,                     mv_set_default(cfg,'dimension_names',{'samples','features','time points'}); end

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% prepare save
if ~iscell(cfg.save), cfg.save = {cfg.save}; end
save_model = any(strcmp(cfg.save, 'model_param'));
save_trainlabel = any(strcmp(cfg.save, 'trainlabel'));

%% Classify across time
if cfg.feedback, mv_print_classification_info(cfg, X, clabel, varargin{:}); end

if ~strcmp(cfg.cv,'none') && ~has_second_dataset

    % Initialize classifier outputs
    cf_output = cell(cfg.repeat, cfg.k, ntime);
    testlabel = cell(cfg.repeat, cfg.k);
    if save_trainlabel, all_trainlabel = cell([cfg.repeat, cfg.k]); end
    if save_model, all_model = cell(size(cf_output)); end

    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end

        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p, cfg.fold, cfg.preprocess, cfg.preprocess_param);

        for kk=1:CV.NumTestSets                     % ---- CV folds ----
            if cfg.feedback
                if kk<=20, fprintf('%d ',kk), % print first 20 folds
                elseif kk==21, fprintf('... ') % then ... and stop to not spam the console too much
                elseif kk>CV.NumTestSets-5, fprintf('%d ',kk) % then the last 5 ones
                end
            end

            % Get train and test data
            [cfg, Xtrain, trainlabel, Xtest, testlabel{rr,kk}] = mv_select_train_and_test_data(cfg, X, clabel, CV.training(kk), CV.test(kk), cfg.is_kernel_matrix);

            if ~isempty(cfg.preprocess)
                % Preprocess train data
                [tmp_cfg, Xtrain, trainlabel] = mv_preprocess(cfg, Xtrain, trainlabel);
                
                % Preprocess test data
                [~, Xtest, testlabel{rr,kk}] = mv_preprocess(tmp_cfg, Xtest, testlabel{rr,kk});
            end
            if save_trainlabel, all_trainlabel{rr,kk} = trainlabel; end

            for tt=1:ntime           % ---- Train and test time ----
                % Train and test data for time point tt
                Xtrain_tt= squeeze(Xtrain(:,:,cfg.time(tt)));
                Xtest_tt= squeeze1(Xtest(:,:,cfg.time(tt)));

                % Train classifier
                cf= train_fun(cfg.hyperparameter, Xtrain_tt, trainlabel);

                % Obtain classifier output (class labels, dvals or probabilities)
                cf_output{rr,kk,tt} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_tt);
                if save_model, all_model{rr,kk,tt} = cf; end
            end
        end
        if cfg.feedback, fprintf('\n'), end
    end

    % Average classification performance across repeats and test folds
    avdim = [1,2];

elseif has_second_dataset
    % -------------------------------------------------------
    % Transfer classification (aka cross decoding) using two datasets. The 
    % first dataset acts as train data, the second as test data.
    X2 = varargin{1};
    assert( (size(X,2)==size(X2,2)) && (size(X,3)==size(X2,3)), sprintf('both datasets must have the same number of features and time points, but size(X) = [%s] and size(X2) = [%s]', num2str(size(X)), num2str(size(X2))))
    
    % Initialize classifier outputs
    cf_output = cell(1, 1, ntime);
    if save_model, all_model = cell(size(cf_output)); end

    % Preprocess train data
    [tmp_cfg, X, clabel] = mv_preprocess(cfg, X, clabel);
    
    % Preprocess test data
    [~, X2, clabel2] = mv_preprocess(tmp_cfg, X2, clabel2);
    
    for tt=1:ntime          % ---- Train and test time ----
        % Train and test data
        Xtrain= squeeze(X(:,:,cfg.time(tt)));
        Xtest= squeeze(X2(:,:,cfg.time(tt)));
        
        % Train classifier
        cf= train_fun(cfg.hyperparameter, Xtrain, clabel);
        
        % Obtain classifier output (class labels or dvals)
        cf_output{1,1,tt} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest);
        if save_model, all_model{tt} = cf; end
    end
    
    all_trainlabel = clabel;
    testlabel = clabel2;
    avdim = [];
else
    % -------------------------------------------------------
    % No cross-validation, just train and test once for each
    % training/testing time. This gives the classification performance for
    % the training set, but it may lead to overfitting and thus to an
    % artifically inflated performance.

    % Initialize classifier outputs
    cf_output = cell(1, 1, ntime);
    if save_model, all_model = cell(size(cf_output)); end

    % Preprocess train/test data
    [~, X, clabel] = mv_preprocess(cfg, X, clabel);

    for tt=1:ntime          % ---- Train and test time ----
        % Train and test data
        Xtraintest= squeeze(X(:,:,cfg.time(tt)));

        % Train classifier
        cf= train_fun(cfg.hyperparameter, Xtraintest, clabel);
        
        % Obtain classifier output (class labels or dvals)
        cf_output{1,1,tt} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtraintest);
        if save_model, all_model{tt} = cf; end
    end

    all_trainlabel = clabel;
    testlabel = clabel;
    avdim = [];
end

%% Calculate performance metrics
if cfg.feedback, fprintf('Calculating performance metrics... '), end
perf = cell(n_metrics, 1);
perf_std = cell(n_metrics, 1);
perf_dimension_names = cell(n_metrics, 1);
for mm=1:n_metrics
    if strcmp(cfg.metric{mm},'none')
        perf{mm} = cf_output;
        perf_std{mm} = [];
        perf_dimension_names{mm} = {'repetition' 'fold' cfg.dimension_names{end}};
    else
        [perf{mm}, perf_std{mm}] = mv_calculate_performance(cfg.metric{mm}, cfg.output_type, cf_output, testlabel, avdim);
        % performance dimension names
        if isvector(perf{mm})
            perf_dimension_names{mm} = cfg.dimension_names(end);
        else
            perf_dimension_names{mm} = [cfg.dimension_names(end) repmat({'metric'}, 1, ndims(perf{mm})-1)];
        end
    end
end
if cfg.feedback, fprintf('finished\n'), end

if n_metrics==1
    perf = perf{1};
    perf_std = perf_std{1};
    perf_dimension_names = perf_dimension_names{1};
    cfg.metric = cfg.metric{1};
end

result = [];
if nargout>1
   result.function              = mfilename;
   result.task                  = 'classification';
   result.perf                  = perf;
   result.perf_std              = perf_std;
   result.metric                = cfg.metric;
   result.perf_dimension_names  = perf_dimension_names;
   result.testlabel             = testlabel;
   result.n                     = size(X,1);
   result.n_metrics             = n_metrics;
   result.n_classes             = n_classes;
   result.classifier            = cfg.classifier;
   result.cfg                   = cfg;
   if save_trainlabel, result.trainlabel = all_trainlabel; end
   if save_model, result.model_param = all_model; end
end