function [perf, result, testlabel] = mv_classify(cfg, X, clabel)
% Flexible classification for arbitrarily sized data.
%
% mv_classify allows for the classification of data of arbitrary number and
% order of dimensions. It combines and generalizes the capabilities of the 
% other high-level functions (mv_crossvalidate, mv_searchlight,
% mv_classify_across_time, mv_classify_timextime).
%
% It is most useful for multi-dimensional datasets such as time-frequency
% data e.g. [samples x channels x frequencies x time points] which do not
% work well with the other high-level functions.
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
% meaning of each dimension. sample_dimension and feature_dimension
% specify which dimension(s) code for samples and features, respectively.
% All other dimensions will be treated as 'searchlight' dimensions and a
% separate classification will be performed for each element of these
% dimensions. Example: let the data matrix be [samples x time x features x
% frequencies]. Let sample_dimension=1 and feature_dimension=3. The output
% of mv_classify will then be a [time x frequencies] (corresponding to
% dimensions 2 and 4) matrix of classification results.
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
% .generalization_dimension - any of the searchlight dimensions can be used
%                             for a generalization. In generalization, a
%                             model is trained for each of the
%                             generalization elements and then tested at
%                             each other element (default []).
% .flatten_features  - if there is multiple feature dimensions, flattens
%                      the features into a vector so that it
%                      can be used with the standard classifiers (default 1)
% .dimension_names   - cell array with names for the dimensions. These names
%                      are be used when printing the classification
%                      info.
%
% CROSS-VALIDATION parameters:
% .cv           - perform cross-validation, can be set to 'kfold',
%                 'leaveout', 'holdout', or 'none' (default 'kfold')
% .k            - number of folds in k-fold cross-validation (default 5)
% .p            - if cv is 'holdout', p is the fraction of test samples
%                 (default 0.1)
% .stratify     search_dim = setdiff(1:ndims(X), [cfg.sample_dimension, cfg.feature_dimension])- if 1, the class proportions are approximately preserved
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
mv_set_default(cfg,'dimension_names',strcat('dim', arrayfun(@(x) {num2str(x)}, 1:ndims(X))));

[cfg, clabel, nclasses, nmetrics] = mv_check_inputs(cfg, X, clabel);

% sort dimensions
cfg.sample_dimension = sort(cfg.sample_dimension);
cfg.feature_dimension = sort(cfg.feature_dimension);
cfg.generalization_dimension = sort(cfg.generalization_dimension);

% define non-sample/feature dimension(s) that will be used for searchlight/looping
search_dim = setdiff(1:ndims(X), [cfg.sample_dimension, cfg.feature_dimension]);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

% indicates whether the data represents kernel matrices
mv_set_default(cfg,'is_kernel_matrix', isfield(cfg.param,'kernel') && strcmp(cfg.param.kernel,'precomputed'));

if cfg.feedback, mv_print_classification_info(cfg,X,clabel); end

%% check dimension parameters

% check sample dimensions
if numel(cfg.sample_dimension) > 2
    error('There can be at most 2 sample dimensions but %d have been specified', numel(cfg.sample_dimension))
elseif (numel(cfg.sample_dimension) == 2) && (~cfg.is_kernel_matrix)
    error('2 sample dimensions given but the kernel is not specified to be precomputed')
end

% check whether dimensions are different and add up to ndims(X)
sam_feat_gen_dims = sort([cfg.sample_dimension, cfg.feature_dimension, cfg.generalization_dimension]);
sam_feat_search_dims = sort([cfg.sample_dimension, cfg.feature_dimension, cfg.searchlight_dimension]);
if numel(unique(sam_feat_gen_dims)) < numel(sam_feat_gen_dims)
    error('sample_dimension, feature_dimension, and searchlight_dimension must be different from each other')
elseif numel(sam_feat_search_dims) ~= ndims(X)
    error('sample_dimension, feature_dimension, and searchlight_dimension together specify %d dimensions but the input data has %d dimensions', numel([cfg.sample_dimension, cfg.feature_dimension, cfg.searchlight_dimension]), ndims(X))
end

if numel(cfg.generalization_dimension) > 1
    error('There should be at most one generalization dimension')
end

%% order the dimensions by samples -> searchlight dimensions -> features

% permute X and dimension names
X = permute(X, [cfg.sample_dimension, cfg.searchlight_dimension, cfg.feature_dimension]);
cfg.dimension_names = cfg.dimension_names([cfg.sample_dimension, cfg.searchlight_dimension, cfg.feature_dimension]);

% adapt the dimensions accordingly
cfg.sample_dimension = 1:numel(cfg.sample_dimension);
cfg.searchlight_dimension = (1:numel(cfg.searchlight_dimension))+cfg.sample_dimension(end);
cfg.feature_dimension = (1:numel(cfg.feature_dimension))+cfg.searchlight_dimension(end);

%% flatten features if necessary
if numel(cfg.feature_dimension) > 1 && cfg.flatten_features
    sz = size(X);
    all_feat = prod(sz(cfg.feature_dimension));
    X = reshape(X, [sz(cfg.sample_dimension), sz(cfg.searchlight_dimension), all_feat]);
    % also flatten dimension names
    cfg.dimension_names{cfg.feature_dimension(1)} = strjoin(cfg.dimension_names(cfg.feature_dimension),'/');
    cfg.dimension_names(cfg.feature_dimension(2:end)) = [];
    cfg.feature_dimension = cfg.sample_dimension+1;
    cfg.searchlight_dimension = (1:numel(cfg.searchlight_dimension)) + cfg.feature_dimension;
end

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

% Define searchlight dimension
sz = size(X);
sz = sz(cfg.searchlight_dimension);
if isempty(sz), sz = 1; end

%% Perform classification
if ~strcmp(cfg.cv,'none') 
    % -------------------------------------------------------
    % Perform cross-validation

    % Initialise classifier outputs
    cf_output = cell([cfg.repeat, cfg.k, sz]);
    testlabel = cell([cfg.repeat, cfg.k]);
    
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
            
            % ---- Generalization ---- (eg time x time)
            % Instead of looping through the generalization dimension, we
            % reshape the data and apply the classifier to all elements at
            % once.
            if any(cfg.generalization_dimension)
                % permute and reshape into [ (trials x test times) x features]
                Xtest= permute(Xtest, [1 3 2]);
                Xtest= reshape(Xtest, numel(testlabel{rr,kk})*nTime2, []);
            end
            
            % ---- Training time ----
            for ss=1:prod(sz)

                % Training data for time point t1
                Xtrain_tt= squeeze(Xtrain(:,:,ss));
                
                % Train classifier
                cf= train_fun(cfg.param, Xtrain_tt, trainlabel);

                % Obtain classifier output (labels, dvals or probabilities)
                cf_output{rr,kk,ss} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), numel(testlabel{rr,kk}),[]);
            end

        end
        if cfg.feedback, fprintf('\n'), end
    end

    % Average classification performance across repeats and test folds
    avdim= [1,2];



elseif strcmp(cfg.cv,'none')
    % -------------------------------------------------------
    % No cross-validation
    
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
    for ss=1:nTime1

        % Training data for time point t1
        Xtrain= squeeze(X(:,:,ss));

        % Train classifier
        cf= train_fun(cfg.param, Xtrain, clabel);

        % Obtain classifier output (labels, dvals or probabilities)
        cf_output{1,1,ss} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), size(X,1),[]);

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

result = [];
if nargout>1
   result.function  = mfilename;
   result.perf      = perf;
   result.perf_std  = perf_std;
   result.metric    = cfg.metric;
   result.cv        = cfg.cv;
   result.k         = cfg.k;
   result.repeat    = cfg.repeat;
   result.nclasses  = nclasses;
   result.classifier = cfg.classifier;
end