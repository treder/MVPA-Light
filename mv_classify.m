function [perf, result, testlabel] = mv_classify(cfg, X, clabel)
% Classification of multi-dimensional data.
%
% mv_classify allows for the classification of data of arbitrary number and
% order of dimensions. It combines and generalizes the capabilities of the 
% other high-level functions (mv_crossvalidate, mv_searchlight,
% mv_classify_across_time, mv_classify_timextime).
%
% It is most useful for multi-dimensional datasets such as time-frequency
% data e.g. [samples x channels x frequencies x time points] which do not
% work well with the other, more specialised high-level functions.
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
% .classifier     - name of classifier, needs to have according train_ and test_
%                   functions (default 'lda')
% .hyperparameter - struct with parameters passed on to the classifier train
%                   function (default [])
% .metric         - classifier performance metric, default 'accuracy'. See
%                   mv_classifier_performance. If set to [] or 'none', the 
%                   raw classifier output (labels, dvals or probabilities 
%                   depending on cfg.output_type) for each sample is returned. 
%                   Use cell array to specify multiple metrics (eg
%                    {'accuracy' 'auc'}
% .feedback       - print feedback on the console (default 1)
%
% For mv_classify to make sense of the data, the user must specify the
% meaning of each dimension. sample_dimension and feature_dimension
% specify which dimension(s) code for samples and features, respectively.
% All other dimensions will be treated as 'search' dimensions and a
% separate classification will be performed for each element of these
% dimensions. Example: let the data matrix be [samples x time x features x
% frequencies]. Let sample_dimension=1 and feature_dimension=3. The output
% of mv_classify will then be a [time x frequencies] (corresponding to
% dimensions 2 and 4) matrix of classification results.
% To use generalization (e.g. time x time, or frequency x frequency), set
% the generalization_dimension parameter. 
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
% .generalization_dimension - any of the other (non-sample, non-feature) 
%                             dimensions can be used for generalization. 
%                             In generalization, a model is trained for each
%                             generalization element and then tested at
%                             each other element (default []). Note: if a 
%                             generalization dimension is given, the input
%                             may not consist of *precomputed kernels*.
%                             This is because the kernel matrix needs to be
%                             evaluated not only between samples within a
%                             given time point but also for all
%                             combinations of samples across different time
%                             points.
% .flatten_features  - if there is multiple feature dimensions, flattens
%                      the features into a single feature vector so that it
%                      can be used with the standard classifiers (default
%                      1). Has no effect if there is only one feature
%                      dimension.
% .dimension_names   - cell array with names for the dimensions. These names
%                      are used when printing the classification
%                      info.
%
% SEARCHLIGHT parameters:
% .neighbours  - [... x ...] matrix specifying which features
%                are neighbours of each other. If there is multiple search
%                dimensions, a cell array of such matrices should be
%                provided. (default: identity matrix). Note: this
%                corresponds to the GRAPH option in mv_searchlight.
%                There is no separate parameter for neighbourhood size, the
%                size of the neighbourhood is specified by the matrix.
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
mv_set_default(cfg,'hyperparameter',[]);
mv_set_default(cfg,'metric','accuracy');
mv_set_default(cfg,'feedback',1);

mv_set_default(cfg,'sample_dimension',1);
mv_set_default(cfg,'feature_dimension',[]);
mv_set_default(cfg,'generalization_dimension',[]);
mv_set_default(cfg,'flatten_features',1);
mv_set_default(cfg,'dimension_names',strcat('dim', arrayfun(@(x) {num2str(x)}, 1:ndims(X))));

mv_set_default(cfg,'neighbours',{});
if isempty(cfg.neighbours), cfg.neighbours = {}; end  % replace [] by {}
if ~iscell(cfg.neighbours), cfg.neighbours = {cfg.neighbours}; end
cfg.neighbours = cfg.neighbours(:);  % make sure it's a column vector

mv_set_default(cfg,'preprocess',{});
mv_set_default(cfg,'preprocess_param',{});

% mv_check_inputs assumes samples are in dimension 1 so need to permute
[cfg, clabel, n_classes, n_metrics] = mv_check_inputs(cfg, permute(X,[cfg.sample_dimension, setdiff(1:ndims(X), cfg.sample_dimension)]), clabel);

% sort dimension vectors
sample_dim = sort(cfg.sample_dimension);
feature_dim = sort(cfg.feature_dimension);
gen_dim = cfg.generalization_dimension;

% define non-sample/feature dimension(s) that will be used for search/looping
search_dim = setdiff(1:ndims(X), [sample_dim, feature_dim]);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:n_classes);

% indicates whether the data represents kernel matrices
mv_set_default(cfg,'is_kernel_matrix', isfield(cfg.hyperparameter,'kernel') && strcmp(cfg.hyperparameter.kernel,'precomputed'));

% generalization does not work together with precomputed kernel matrices
if cfg.is_kernel_matrix && ~isempty(gen_dim)
    error('generalization does not work together with precomputed kernel matrices')
end

if cfg.feedback, mv_print_classification_info(cfg,X,clabel); end

%% check dimension parameters
% check sample dimensions
if numel(sample_dim) > 2
    error('There can be at most 2 sample dimensions but %d have been specified', numel(sample_dim))
elseif (numel(sample_dim) == 2) && (~cfg.is_kernel_matrix)
    error('there is 2 sample dimensions given but the kernel is not specified to be precomputed (set cfg.hyperparameter.kernel=''precomputed'')')
elseif numel(sample_dim) == 2  &&  numel(feature_dim)>1
    error('if there is 2 samples dimensions you must set cfg.feature_dimensions=[]')
elseif numel(gen_dim) > 1
    % check generalization dimensions
    error('There can be at most one generalization dimension')
end

% check whether dimensions are different and add up to ndims(X)
sam_feat_gen_dims = sort([sample_dim, feature_dim, gen_dim]);
if numel(unique(sam_feat_gen_dims)) < numel(sam_feat_gen_dims)
    error('sample_dimension, feature_dimension, and generalization_dimension must be different from each other')
end

%% check neighbours parameters
has_neighbours = ~isempty(cfg.neighbours);

if has_neighbours && (numel(cfg.neighbours) ~= numel(search_dim))
    error('If any neighbourhood matrix is given, you must specify a matrix for every search dimension')
end

if has_neighbours && numel(gen_dim)>0
    error('Searchlight and generalization are currently not supported simultaneously')
end
%% order the dimensions by samples -> search dimensions -> features

% the generalization dimension should be the  last of the search dimensions,
% if it is not then permute the dimensions accordingly
if ~isempty(gen_dim) && (search_dim(end) ~= gen_dim)
    ix = find(ismember(search_dim, gen_dim));
    % push gen dim to the end
    search_dim = [search_dim(1:ix-1), search_dim(ix+1:end), search_dim(ix)];
    % use circshift to push dimension to the end
%     search_dim = circshift(search_dim(, numel(search_dim)-ix);
end

% permute X and dimension names
X = permute(X, [sample_dim, search_dim, feature_dim]);
cfg.dimension_names = cfg.dimension_names([sample_dim, search_dim, feature_dim]);

% adapt the dimensions to reflect the permuted X
sample_dim = 1:numel(sample_dim);
search_dim = (1:numel(search_dim))  + numel(sample_dim);
feature_dim = (1:numel(feature_dim))+ numel(sample_dim) + numel(search_dim);
if ~isempty(gen_dim), gen_dim = search_dim(end); end

%% flatten features to one dimension if requested
if numel(feature_dim) > 1 && cfg.flatten_features
    sz_search = size(X);
    all_feat = prod(sz_search(feature_dim));
    X = reshape(X, [sz_search(sample_dim), sz_search(search_dim), all_feat]);
    % also flatten dimension names
    cfg.dimension_names{feature_dim(1)} = strjoin(cfg.dimension_names(feature_dim),'/');
    cfg.dimension_names(feature_dim(2:end)) = [];
    feature_dim = feature_dim(1);
end

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

% Define search dimension
sz_search = size(X);
sz_search = sz_search(search_dim);
if isempty(sz_search), sz_search = 1; end

% sample_skip and feature_skip helps us access the search dimensions by 
% skipping over sample and feature dimensions
% sample_skip = repmat({':'},[1, numel([sample_dim, feature_dim])] );
sample_skip = repmat({':'},[1, numel(sample_dim)] );
feature_skip = repmat({':'},[1, numel(feature_dim)] );

%% Create all combinations of elements in the search dimensions
if isempty(search_dim)
    % no search dimensions, we just perform cross-validation once
    dim_loop = {':'};
else
    len_loop = prod(sz_search);
    dim_loop = zeros(numel(sz_search), len_loop);
    for rr = 1:numel(sz_search)  % row
        seq = repelem(1:sz_search(rr), prod(sz_search(1:rr-1)));
        dim_loop(rr, :) = repmat(seq, [1, len_loop/numel(seq)]);
    end
    
    % to use dim_loop for indexing, we need to convert it to a cell array
    dim_loop = num2cell(dim_loop);
end

nfeat = size(X);
nfeat = nfeat(feature_dim);
if isempty(nfeat), nfeat = 1; end

%% Perform classification
if ~strcmp(cfg.cv,'none') 
    % -------------------------------------------------------
    % Perform cross-validation

    % Initialise classifier outputs
    cf_output = cell([cfg.repeat, cfg.k, sz_search]);
    testlabel = cell([cfg.repeat, cfg.k]);
    
    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end
        
        % Define cross-validation
        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p, cfg.fold);
        
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
            
            if ~isempty(gen_dim)
                % ---- Generalization ---- (eg time x time)
                % Instead of looping through the generalization dimension,
                % which would require an additional loop, we reshape the test
                % data and apply the classifier to all elements of the
                % generalization dimension at once
                
                % gen_dim is the last search dimension. For reshaping we
                % need to move it to the first search dim position and
                % shift the other dimensions up one position, we can use 
                % circshift for this
                Xtest = permute(Xtest, [sample_dim, circshift(search_dim,1), feature_dim]);
                
                % reshape samples x gen dim into one dimension
                sz_search = size(Xtest);
                Xtest = reshape(Xtest, [sz_search(1)*sz_search(2), sz_search(3:end)]);
            end
            
            % Remember sizes
            sz_Xtrain = size(Xtrain);
            sz_Xtest = size(Xtest);
            
            for ix = dim_loop                       % ---- search dimensions ----
                                
                % Training data for current search position
                if has_neighbours
                    % --- searchlight --- define neighbours for current iteration
                    ix_nb = cellfun( @(N,f) find(N(f,:)), cfg.neighbours, ix, 'Un',0);
                    % train data
                    X_ix = Xtrain(sample_skip{:}, ix_nb{:}, feature_skip{:});
                    X_ix = reshape(X_ix, [sz_Xtrain(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
                    % test data
                    Xtest_ix = squeeze(Xtest(sample_skip{:}, ix_nb{:}, feature_skip{:}));
                    Xtest_ix = reshape(Xtest_ix, [sz_Xtest(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
                else
                    if isempty(gen_dim),    ix_test = ix;
                    else,                   ix_test = ix(1:end-1);
                    end
                    X_ix = squeeze(Xtrain(sample_skip{:}, ix{:}, feature_skip{:}));
                    Xtest_ix = squeeze(Xtest(sample_skip{:}, ix_test{:}, feature_skip{:}));
                end
                
                % Train classifier
                cf= train_fun(cfg.hyperparameter, X_ix, trainlabel);

                % Obtain classifier output (labels, dvals or probabilities)
                if isempty(gen_dim)
                    cf_output{rr,kk,ix{:}} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix);
                else
                    % we have to reshape classifier output back
                    cf_output{rr,kk,ix{:}} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix), numel(testlabel{rr,kk}),[]);
                end
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
    if ~isempty(cfg.preprocess)
        [~, X, clabel] = mv_preprocess(cfg, X, clabel);
    end
    
    % Initialise classifier outputs
    cf_output = cell([1, 1, sz_search]);
    
    if ~isempty(gen_dim)
        Xtest= permute(X, [sample_dim, circshift(search_dim,1), feature_dim]);
        
        % reshape samples x gen dim into one dimension
        sz_search = size(Xtest);
        Xtest= reshape(Xtest, [sz_search(1)*sz_search(2), sz_search(3:end)]);
    else
        Xtest = X;
    end
    
    % Remember sizes
    sz_Xtrain = size(X);
    sz_Xtest = size(Xtest);
    
    for ix = dim_loop                       % ---- search dimensions ----
        
        % Training data for current search position
        if has_neighbours
            % --- searchlight --- define neighbours for current iteration
            ix_nb = cellfun( @(N,f) find(N(f,:)), cfg.neighbours, ix, 'Un',0);
            % train data
            X_ix = X(sample_skip{:}, ix_nb{:}, feature_skip{:});
            X_ix= reshape(X_ix, [sz_Xtrain(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
            % test data
            Xtest_ix = squeeze(Xtest(sample_skip{:}, ix_nb{:}, feature_skip{:}));
            Xtest_ix = reshape(Xtest_ix, [sz_Xtest(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
        else
            if isempty(gen_dim),    ix_test = ix;
            else,                   ix_test = ix(1:end-1);
            end
            X_ix= squeeze(X(sample_skip{:}, ix{:}, feature_skip{:}));
            Xtest_ix = squeeze(Xtest(sample_skip{:}, ix_test{:}, feature_skip{:}));
        end
        
        % Train classifier
        cf= train_fun(cfg.hyperparameter, X_ix, clabel);
        
        % Obtain classifier output (labels, dvals or probabilities)
        if isempty(gen_dim)
            cf_output{1,1,ix{:}} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix);
        else
            % we have to reshape classifier output back
            cf_output{1,1,ix{:}} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix), numel(clabel),[]);
        end
    end

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
    else
        [perf{mm}, perf_std{mm}] = mv_calculate_performance(cfg.metric{mm}, cfg.output_type, cf_output, testlabel, avdim);
        % performance dimension names
        if isvector(perf{mm})
            perf_dimension_names{mm} = cfg.dimension_names(search_dim);
        else
            if ~isempty(gen_dim)
                ix_gen_in_search_dim = find(search_dim == gen_dim);
                names = cfg.dimension_names(search_dim);
                names{ix_gen_in_search_dim} = ['train ' names{ix_gen_in_search_dim}];
                perf_dimension_names{mm} = [names repmat({'metric'}, 1, ndims(perf{mm})-numel(search_dim)-numel(gen_dim)) {['test ' cfg.dimension_names{gen_dim}]}];
            else
                perf_dimension_names{mm} = [cfg.dimension_names(search_dim) repmat({'metric'}, 1, ndims(perf{mm})-numel(search_dim)-numel(gen_dim)) cfg.dimension_names(gen_dim)];
            end
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
   result.perf_dimension_names  = perf_dimension_names;
   result.metric                = cfg.metric;
   result.n                     = size(X, 1);
   result.n_metrics             = n_metrics;
   result.n_classes             = n_classes;
   result.classifier            = cfg.classifier;
   result.cfg                   = cfg;
end