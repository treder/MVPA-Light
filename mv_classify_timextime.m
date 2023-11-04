function [perf, result, testlabel] = mv_classify_timextime(cfg, X, clabel, varargin)
% Time x time generalisation. A classifier is trained on the training data
% X and validated on either the same dataset X. Cross-validation is
% recommended to avoid overfitting. If another dataset X2 is provided,
% the classifier is trained on X and tested on X2. No cross-validation is
% performed in this case since the datasets are assumed to be independent.
%
% Note that this function does not work with *precomputed* kernels since
% they require the kernel matrix to be evaluated for different combinations 
% of time points as well.
%
% Usage:
% perf = mv_classify_timextime(cfg, X, clabel, <X2, clabel2>)
%
%Parameters:
% X              - [samples x features x time points] data matrix. 
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
%                 function (default []). 
% .metric       - classifier performance metric, default 'accuracy'. See
%                 mv_classifier_performance. If set to [] or 'none', the 
%                 raw classifier output (labels, dvals or probabilities 
%                 depending on cfg.output_type) for each sample is returned. 
%                 Use cell array to specify multiple metrics (eg
%                 {'accuracy' 'auc'}
% .time1        - indices of training time points (by default all time
%                 points in X are used)
% .time2        - indices of test time points (by default all time points
%                 are used)
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
% .p            - if cv='holdout', p is the fraction of test samples
%                 (default 0.1)
% .stratify     - if 1, the class proportions are approximately preserved
%                 in each fold (default 1)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds (default 1)
% .fold         - if cv='predefined', fold is a vector of length
%                 #samples that specifies the fold each sample belongs to
%
% PREPROCESSING parameters: and 
% .preprocess         - cell array containing the preprocessing pipeline. The
%                       pipeline is applied in chronological order
% .preprocess_param   - cell array of preprocessing parameter structs for each
%                       function. Length of preprocess_param must match length
%                       of preprocess
%
% Returns:
% perf          - time1 x time2 classification matrix of classification
%                 performances corresponding to the selected metric. If
%                 metric='none', perf is a [r x k x t] cell array of
%                 classifier outputs, where each cell corresponds to a test
%                 set, k is the number of folds, r is the number of 
%                 repetitions, and t is the number of training time points.
%                 Each cell contains [n x t2] elements, where n is the
%                 number of test samples and t2 is the number of test time
%                 points. If multiple metrics are requested, perf is a cell array
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
mv_set_default(cfg,'time1',1:size(X,3));
mv_set_default(cfg,'feedback',1);
mv_set_default(cfg,'save',{});

mv_set_default(cfg,'sample_dimension',1);
mv_set_default(cfg,'dimension_names',{'samples','features','time points'});
mv_set_default(cfg,'preprocess',{});
mv_set_default(cfg,'preprocess_param',{});

[cfg, clabel, n_classes, n_metrics, clabel2] = mv_check_inputs(cfg, X, clabel, varargin{:});

has_second_dataset = (nargin==5);
if has_second_dataset
    X2 = double(varargin{1});
    mv_set_default(cfg,'time2',1:size(X2,3));
else
    mv_set_default(cfg,'time2',1:size(X,3));
end
ntime1 = numel(cfg.time1);
ntime2 = numel(cfg.time2);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:n_classes);

% this function does not work with precomputed kernel matrices
if isfield(cfg.hyperparameter,'kernel') && strcmp(cfg.hyperparameter.kernel,'precomputed')
    error('mv_classify_timextime does not work with precomputed kernel matrices, since kernel needs to be evaluated between time points as well')
end

%% Reduce data to selected time points
X = X(:,:,cfg.time1);

if has_second_dataset
    X2 = X2(:,:,cfg.time2);
end

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% prepare save
if ~iscell(cfg.save), cfg.save = {cfg.save}; end
save_model = any(strcmp(cfg.save, 'model_param'));
save_trainlabel = any(strcmp(cfg.save, 'trainlabel'));

%% Time x time generalisation
if cfg.feedback, mv_print_classification_info(cfg, X, clabel, varargin{:}); end

if ~strcmp(cfg.cv,'none') && ~has_second_dataset
    % -------------------------------------------------------
    % One dataset X has been provided as input. X is hence used for both
    % training and testing. To avoid overfitting, cross-validation is
    % performed.

    % Initialize classifier outputs
    cf_output = cell(cfg.repeat, cfg.k, ntime1);
    testlabel = cell(cfg.repeat, cfg.k);
    if save_trainlabel, all_trainlabel = cell([cfg.repeat, cfg.k]); end
    if save_model, all_model = cell(size(cf_output)); end

    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end
        
        % Define cross-validation
        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p, cfg.fold, cfg.preprocess, cfg.preprocess_param);
        
        for kk=1:CV.NumTestSets                      % ---- CV folds ----
            if cfg.feedback
                if kk<=20, fprintf('%d ',kk), % print first 20 folds
                elseif kk==21, fprintf('... ') % then ... and stop to not spam the console too much
                elseif kk>CV.NumTestSets-5, fprintf('%d ',kk) % then the last 5 ones
                end
            end

            % Get train and test data
            [cfg, Xtrain, trainlabel, Xtest, testlabel{rr,kk}] = mv_select_train_and_test_data(cfg, X, clabel, CV.training(kk), CV.test(kk), 0);

            if ~isempty(cfg.preprocess)
                % Preprocess train data
                [tmp_cfg, Xtrain, trainlabel] = mv_preprocess(cfg, Xtrain, trainlabel);
                
                % Preprocess test data
                [~, Xtest, testlabel{rr,kk}] = mv_preprocess(tmp_cfg, Xtest, testlabel{rr,kk});
            end
            if save_trainlabel, all_trainlabel{rr,kk} = trainlabel; end
            
            % ---- Test data ----
            % Instead of looping through the second time dimension, we
            % reshape the data and apply the classifier to all time
            % points. We then need to apply the classifier only once
            % instead of nTime2 times.

            % permute and reshape into [ (trials x test times) x features]
            % samples
            Xtest= permute(Xtest, [1 3 2]);
            Xtest= reshape(Xtest, numel(testlabel{rr,kk})*ntime2, []);

            % ---- Training time ----
            for t1=1:ntime1

                % Training data for time point t1
                Xtrain_tt= squeeze(Xtrain(:,:,t1));
                
                % Train classifier
                cf= train_fun(cfg.hyperparameter, Xtrain_tt, trainlabel);

                % Obtain classifier output (labels, dvals or probabilities)
                cf_output{rr,kk,t1} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), numel(testlabel{rr,kk}),[]);
                if save_model, all_model{rr,kk,t1} = cf; end
            end

        end
        if cfg.feedback, fprintf('\n'), end
    end

    % Average classification performance across repeats and test folds
    avdim= [1,2];

elseif has_second_dataset
    % -------------------------------------------------------
    % Transfer classification (aka cross decoding) using two datasets. The 
    % first dataset acts as train data, the second as test data.
    assert( size(X,2)==size(X2,2), sprintf('both datasets must have the same number of features, but size(X) = [%s] and size(X2) = [%s]', num2str(size(X)), num2str(size(X2))))

    % Initialize classifier outputs
    cf_output = cell(1, 1, ntime1);
    if save_model, all_model = cell(size(cf_output)); end

    % Preprocess train data
    [tmp_cfg, X, clabel] = mv_preprocess(cfg, X, clabel);
    
    % Preprocess test data
    [~, X2, clabel2] = mv_preprocess(tmp_cfg, X2, clabel2);

    % Permute and reshape into [ (trials x test times) x features]
    Xtest= permute(X2, [1 3 2]);
    Xtest= reshape(Xtest, size(X2,1)*ntime2, []);

    % ---- Training time ----
    for t1=1:ntime1

        % Training data for time point t1
        Xtrain= squeeze(X(:,:,t1));

        % Train classifier
        cf= train_fun(cfg.hyperparameter, Xtrain, clabel);

        % Obtain classifier output (labels or dvals)
        cf_output{1,1,t1} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), size(X2,1),[]);
        if save_model, all_model{t1} = cf; end
    end

    all_trainlabel = clabel;
    testlabel = clabel2;
    avdim = [];

elseif strcmp(cfg.cv,'none')
    % -------------------------------------------------------
    % No cross-validation, just train and test once for each
    % training/testing time. This gives the classification performance for
    % the training set, but it may lead to overfitting and thus to an
    % artifically inflated performance.
    
    % Initialize classifier outputs
    cf_output = cell(1, 1, ntime1);
    if save_model, all_model = cell(size(cf_output)); end
    
    % Preprocess train/test data
    [~, X, clabel] = mv_preprocess(cfg, X, clabel);

    % Permute and reshape into [ (trials x test times) x features]
    Xtest= permute(X, [1 3 2]);
    Xtest= reshape(Xtest, size(X,1)*ntime1, []);
    % permute and reshape into [ (trials x test times) x features]
  
    % ---- Training time ----
    for t1=1:ntime1

        % Training data for time point t1
        Xtrain= squeeze(X(:,:,t1));

        % Train classifier
        cf= train_fun(cfg.hyperparameter, Xtrain, clabel);

        % Obtain classifier output (labels, dvals or probabilities)
        cf_output{1,1,t1} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest), size(X,1),[]);
        if save_model, all_model{t1} = cf; end
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
        perf_dimension_names{mm} = [{['train ' cfg.dimension_names{end}]} repmat({'metric'}, 1, ndims(perf{mm})-2) {['test ' cfg.dimension_names{end}]}];
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
   if has_second_dataset
       result.n                 = size(X2,1);
   else
       result.n                 = size(X,1);
   end
   result.n_metrics             = n_metrics;
   result.n_classes             = n_classes;
   result.classifier            = cfg.classifier;
   result.cfg                   = cfg;
   if save_trainlabel, result.trainlabel = all_trainlabel; end
   if save_model, result.model_param = all_model; end
end