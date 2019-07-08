function [perf, result, testlabel] = mv_crossvalidate(cfg, X, clabel)
% Cross-validation. A classifier is trained and validated for
% given 2D [samples x features] dataset X.
%
% If the data has a time component as well [samples x features x time] and
% is 3D, mv_classify_across_time can be used instead to perform
% cross-validated classification for each time point. mv_classify_timextime
% can be used for time generalisation.
%
% Usage:
% [perf, res] = mv_crossvalidate(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] data matrix -OR-
%                  [samples x samples] kernel matrix
% clabel         - [samples x 1] vector of class labels
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
% .feedback     - print feedback on the console (default 1)
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
%
% Returns:
% perf          - classifier performance corresponding to the selected
%                 metric. If metric='none', perf is a r x k cell array of
%                 classifier outputs, where each cell corresponds to a test
%                 set, k is the number of folds and r is the number of 
%                 repetitions. If multiple metrics are requested, perf is a cell array
% result        - struct with fields describing the classification result.
%                 Can be used as input to mv_statistics and mv_plot_result
% testlabel     - r x k cell array of test labels. Can be useful if
%                 metric='none'
% 

% (c) Matthias Treder

X = double(X);
if ~ismatrix(X), error('X must be 2-dimensional'), end

mv_set_default(cfg,'classifier','lda');
mv_set_default(cfg,'param',[]);
mv_set_default(cfg,'metric','accuracy');
mv_set_default(cfg,'feedback',1);

mv_set_default(cfg,'sample_dimension',1);
mv_set_default(cfg,'preprocess',{});
mv_set_default(cfg,'preprocess_param',{});

[cfg, clabel, nclasses, nmetrics] = mv_check_inputs(cfg, X, clabel);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

% indicates whether the data represents kernel matrices
mv_set_default(cfg,'is_kernel_matrix', isfield(cfg.param,'kernel') && strcmp(cfg.param.kernel,'precomputed'));

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% Cross-validate
if cfg.feedback, mv_print_classification_info(cfg,X,clabel); end

if ~strcmp(cfg.cv,'none')

    % Initialise classifier outputs
    cf_output = cell(cfg.repeat, cfg.k);
    testlabel = cell(cfg.repeat, cfg.k);

    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end

        % Define cross-validation
        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p);

        for kk=1:CV.NumTestSets                     % ---- CV folds ----
            if cfg.feedback, fprintf('%d ',kk), end

            % Get train and test data
            [Xtrain, trainlabel, Xtest, testlabel{rr,kk}] = mv_select_train_and_test_data(X, clabel, CV.training(kk), CV.test(kk), cfg.is_kernel_matrix);

            if ~isempty(cfg.preprocess)
                % Preprocess train data
                [tmp_cfg, Xtrain, trainlabel] = mv_preprocess(cfg, Xtrain, trainlabel);
                
                % Preprocess test data
                [~, Xtest, testlabel{rr,kk}] = mv_preprocess(tmp_cfg, Xtest, testlabel{rr,kk});
            end
            
            % Train classifier on training data
            cf= train_fun(cfg.param, Xtrain, trainlabel);

            % Obtain classifier output (labels, dvals or probabilities) on test data
            cf_output{rr,kk} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest);

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

    % Preprocess train/test data
    [~, X, clabel] = mv_preprocess(cfg, X, clabel);

    % Train classifier
    cf= train_fun(cfg.param, X, clabel);

    % Obtain classifier output (labels, dvals or probabilities)
    cf_output = mv_get_classifier_output(cfg.output_type, cf, test_fun, X);

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
% if isempty(cfg.metric) || strcmp(cfg.metric,'none')
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