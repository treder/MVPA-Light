function [perf, result, testlabel] = mv_classify(cfg, X, clabel, X2, clabel2)
% Flexible classification analysis for arbitrarily sized data.
%
% mv_classify allows for the classification of data of arbitrary number and
% order of dimensions. It combines and generalizes the capabilities of the 
% other high-level functions (mv_crossvalidate, mv_searchlight,
% mv_classify_across_time, mv_classify_timextime).
%
% It is most useful for multi-dimensional datasets such as time-frequency
% data e.g. [samples x channels x frequencies x time points] which does not
% work with the other high-level functions.
%
% Usage:
% [perf, res] = mv_classify(cfg, X, clabel)
%
%Parameters:
% X              - [... x ... x ... x ] data matrix or kernel matrix of
%                  arbitrary dimensions
% clabel         - [samples x 1] vector of class labels
%
% cfg          - struct with optional parameters:
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
% For mv_classify to make sense of the data, the user must specify the
% meaning of each dimension. Sample, feature, and searchlight dimensions
% must be different and together they must specify all dimensions of the
% data matrix.
% 
% .sample_dimension  - the data dimension(s) that code the samples (default 1). 
%                      It has either one element or two elements when the
%                      data provided is a kernel matrix. 
%                      It cannot have more than 2 elements.
% .feature_dimension - the data dimension(s) that code the features (default
%                      2). There can be more than 1 feature dimension, but
%                      then a classifier must be used that can deal with
%                      multi-dimensional inputs. If a kernel matrix is
%                      provided, there cannot be a feature dimension.
% .searchlight_dimension - the dimension(s) across which the loops are
%                          performed. A separate classification is
%                          performed for each of the elements of the
%                          searchlight dimensions  (default []).
% .generalization_dimension - any of the searchlight dimensions can be used
%                             for a generalization. In generalization, a
%                             model is trained for each of the
%                             generalization elements and then tested at
%                             each other element (default []).
% .flatten_features  - if there is more than 1 feature dimension, flattens
%                      the feature matrix/array into a vector so that it
%                      can be used with the standard classifiers (default 1)
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
% PREPROCESSING parameters:  
% .preprocess         - cell array containing the preprocessing pipeline. The
%                       pipeline is applied in chronological order
% .preprocess_param   - cell array of preprocessing parameter structs for each
%                       function. Length of preprocess_param must match length
%                       of preprocess
%
% Returns:
% perf          - matrix of classification performances corresponding to 
%                 the selected metric. If multiple metrics are requested, 
%                 perf is a cell array
% result        - struct with fields describing the classification result.
%                 Can be used as input to mv_statistics and mv_plot_result
% testlabel     - cell array of test labels. Can be useful if metric='none'

% (c) matthias treder

X = double(X);

mv_set_default(cfg,'classifier','lda');
mv_set_default(cfg,'param',[]);
mv_set_default(cfg,'metric','accuracy');
mv_set_default(cfg,'feedback',1);

mv_set_default(cfg,'preprocess',{});
mv_set_default(cfg,'preprocess_param',{});

mv_set_default(cfg,'sample_dimension',1);
mv_set_default(cfg,'feature_dimension',2);
mv_set_default(cfg,'searchlight_dimension',[]);
mv_set_default(cfg,'generalization_dimension',[]);
mv_set_default(cfg,'flatten_features',1);

[cfg, clabel, nclasses, nmetrics] = mv_check_inputs(cfg, X, clabel);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

% indicates whether the data represents kernel matrices
mv_set_default(cfg,'is_kernel_matrix', isfield(cfg.param,'kernel') && strcmp(cfg.param.kernel,'precomputed'));

%% check dimension parameters
if numel(cfg.sample_dimension) > 2
    error('There can be at most 2 sample dimensions but %d have been specified', numel(cfg.sample_dimension))
elseif (numel(cfg.sample_dimension) == 2) && (~cfg.is_kernel_matrix)
    error('2 sample dimensions given but the kernel is not specified to be precomputed')
end

if any([intersect(cfg.sample_dimension, cfg.feature_dimension), intersect(cfg.sample_dimension, cfg.searchlight_dimension), intersect(cfg.searchlight_dimension, cfg.feature_dimension)])
    error('sample_dimension, feature_dimension, and searchlight_dimension must be different from each other')
elseif numel([cfg.sample_dimension, cfg.feature_dimension, cfg.searchlight_dimension]) ~= ndims(X)
    error('sample_dimension, feature_dimension, and searchlight_dimension together specify %d dimensions but the input data has %d dimensions', numel([cfg.sample_dimension, cfg.feature_dimension, cfg.searchlight_dimension]), ndims(X))
end

%% order the dimensions by samples -> features -> searchlight dimensions
X = permute(X, [cfg.sample_dimension, cfg.feature_dimension, cfg.searchlight_dimension]);
cfg.sample_dimension = 1:numel(cfg.sample_dimension);
cfg.feature_dimension = (1:numel(cfg.feature_dimension))+cfg.sample_dimension(end);
cfg.searchlight_dimension = (1:numel(cfg.searchlight_dimension))+cfg.feature_dimension(end);

%% flatten features if necessary
if numel(cfg.feature_dimension) > 1 && cfg.flatten_features
    sz = size(X);
    all_feat = prod(cfg.feature_dimension);
    X = reshape(X, [sz(cfg.sample_dimension), all_feat, sz(cfg.searchlight_dimension)]);
    cfg.feature_dimension = cfg.sample_dimension+1;
    cfg.searchlight_dimension = (1:numel(cfg.searchlight_dimension)) + cfg.feature_dimension;
end

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);


%% --- todo --- hier weiter --- 

%% Perform classification
if ~strcmp(cfg.cv,'none') 
    % -------------------------------------------------------
    % One dataset X has been provided as input. X is hence used for both
    % training and testing. To avoid overfitting, cross-validation is
    % performed.
    if cfg.feedback, mv_print_classification_info(cfg,X,clabel); end

    % Initialise classifier outputs
    cf_output = cell(cfg.repeat, cfg.k, nTime1);
    testlabel = cell(cfg.repeat, cfg.k);
    
    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end
        
        % Define cross-validation
        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p);
        
        for kk=1:CV.NumTestSets                      % ---- CV folds ----
            if cfg.feedback, fprintf('%d ',kk), end

            % Get train and test data
            [Xtrain, trainlabel, Xtest, testlabel{rr,kk}] = mv_select_train_and_test_data(X, clabel, CV.training(kk), CV.test(kk), cfg.is_kernel_matrix);

            if ~isempty(cfg.preprocess)
                % Preprocess train data
                [tmp_cfg, Xtrain, trainlabel] = mv_preprocess(cfg, Xtrain, trainlabel);
                
                % Preprocess test data
                [~, Xtest, testlabel{rr,kk}] = mv_preprocess(tmp_cfg, Xtest, testlabel{rr,kk});
            end
            
            % ---- Test data ----
            % Instead of looping through the second time dimension, we
            % reshape the data and apply the classifier to all time
            % points. We then need to apply the classifier only once
            % instead of nTime2 times.

            % permute and reshape into [ (trials x test times) x features]
            Xtest= permute(Xtest, [1 3 2]);
            Xtest= reshape(Xtest, numel(testlabel{rr,kk})*nTime2, []);

            % ---- Training time ----
            for t1=1:nTime1

                % Training data for time point t1
                Xtrain_tt= squeeze(Xtrain(:,:,t1));
                
                % Train classifier
                cf= train_fun(cfg.param, Xtrain_tt, trainlabel);

                % Obtain classifier output (labels, dvals or probabilities)
                cf_output{rr,kk,t1} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), numel(testlabel{rr,kk}),[]);
            end

        end
        if cfg.feedback, fprintf('\n'), end
    end

    % Average classification performance across repeats and test folds
    avdim= [1,2];

elseif hasX2
    % -------------------------------------------------------
    % An additional dataset X2 has been provided. The classifier is trained
    % on X and tested on X2. Cross-validation does not make sense here and
    % is not performed.
    cfg.cv = 'none';
    
    % Print info on datasets
    if cfg.feedback, mv_print_classification_info(cfg, X, clabel, X2, clabel2); end
    
    % Preprocess train data
    [tmp_cfg, X, clabel] = mv_preprocess(cfg, X, clabel);
    
    % Preprocess test data
    [~, X2, clabel2] = mv_preprocess(tmp_cfg, X2, clabel2);

    % Initialise classifier outputs
    cf_output = cell(1, 1, nTime1);

    % permute and reshape into [ (trials x test times) x features]
    Xtest= permute(X2, [1 3 2]);
    Xtest= reshape(Xtest, size(X2,1)*nTime2, []);

    % ---- Training time ----
    for t1=1:nTime1

        % Training data for time point t1
        Xtrain= squeeze(X(:,:,t1));

        % Train classifier
        cf= train_fun(cfg.param, Xtrain, clabel);

        % Obtain classifier output (labels or dvals)
        cf_output{1,1,t1} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), size(X2,1),[]);

    end

    testlabel = clabel2;
    avdim = [];

elseif strcmp(cfg.cv,'none')
    % -------------------------------------------------------
    % One dataset X has been provided as input. X is hence used for both
    % training and testing. However, cross-validation is not performed.
    % Note that this can lead to overfitting.

    if cfg.feedback
        fprintf('Training and testing on the same dataset (note: this can lead to overfitting).\n')
    end
    
    % Preprocess train/test data
    [~, X, clabel] = mv_preprocess(cfg, X, clabel);

    % Initialise classifier outputs
    cf_output = cell(1, 1, nTime1);

    % permute and reshape into [ (trials x test times) x features]
    Xtest= permute(X, [1 3 2]);
    Xtest= reshape(Xtest, size(X,1)*nTime1, []);
    % permute and reshape into [ (trials x test times) x features]
  
    % ---- Training time ----
    for t1=1:nTime1

        % Training data for time point t1
        Xtrain= squeeze(X(:,:,t1));

        % Train classifier
        cf= train_fun(cfg.param, Xtrain, clabel);

        % Obtain classifier output (labels, dvals or probabilities)
        cf_output{1,1,t1} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), size(X,1),[]);

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
   if hasX2
       result.n         = size(X2,1);
   else
       result.n         = size(X,1);
   end
   result.repeat    = cfg.repeat;
   result.nclasses  = nclasses;
   result.classifier = cfg.classifier;
end