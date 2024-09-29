function [cfg, clabel, n_classes,n_metrics, clabel2] = mv_check_inputs(cfg, X, clabel, X2, clabel2)
% Performs some sanity checks and sets some defaults for input parameters 
% cfg, X, and y.
% Also checks whether external toolboxes (LIBSVM and LIBLINEAR) are
% available if required.

if ~iscell(cfg.metric)
    cfg.metric = {cfg.metric};
end
n_metrics = numel(cfg.metric);

has_second_dataset = (nargin > 3) && ~isempty(X2) && ~isempty(clabel2);
if ~has_second_dataset, clabel2 = []; end

%% clabel: check class labels
clabel = clabel(:);
u = unique(clabel);
n_classes = length(u);

if n_classes==1
    error('Only one class specified. Class labels must contain at least 2 classes')
end

if iscell(clabel) || ~all(ismember(clabel,1:n_classes))
    warning('clabel should be a vector consisting of integers 1 (class 1), 2 (class 2), 3 (class 3) and so on. Relabelling them accordingly.');
    newlabel = nan(numel(clabel), 1);
    for i = 1:n_classes
        newlabel(ismember(clabel, u(i))) = i; % set to 1:nth classes
    end
    clabel = newlabel;
    if has_second_dataset
        newlabel = nan(numel(clabel2), 1);
        for i = 1:n_classes
            newlabel(ismember(clabel2, u(i))) = i;
        end
        clabel2 = newlabel;
    end
end

%% clabel and cfg: check whether there's more than 2 classes but yet a binary classifier is used
binary_classifiers = {'lda' 'logreg' 'svm'};
if n_classes > 2 && ismember(cfg.classifier, binary_classifiers)
    error('Cannot use %s for a classification task with %d classes: use a multiclass classifier instead (see https://github.com/treder/MVPA-Light/ for a list of classifiers)', upper(cfg.classifier), n_classes)
end

%% cfg: check whether all parameters are written in lowercase
fn = fieldnames(cfg);

% Are all cfg fields given in lowercase?
not_lowercase = find(~strcmp(fn,lower(fn)));
if any(not_lowercase)
    error('For consistency, all parameters must be given in lowercase: please replace cfg.%s by cfg.%s', fn{not_lowercase(1)},lower(fn{not_lowercase(1)}) )
end

% Are all cfg.hyperparameter fields given in lowercase?
if isfield(cfg,'hyperparameter') && isstruct(cfg.hyperparameter)
    pfn = fieldnames(cfg.hyperparameter);
    not_lowercase = find(~strcmp(pfn,lower(pfn)));
    
    if any(not_lowercase)
        error('For consistency, all parameters must be given in lowercase: please replace hyperparameter.%s by hyperparameter.%s', pfn{not_lowercase(1)},lower(pfn{not_lowercase(1)}) )
    end
end

%% cfg: given a metric, set default for output_type
if any(ismember({'dval','auc','roc','tval'},cfg.metric))
    mv_set_default(cfg,'output_type','dval');
else
    mv_set_default(cfg,'output_type','clabel');
end

%% cfg: check whether number of classes and metric are compatible (eg 'auc' does not work for more than 2 classes)
only_binary = {'auc' 'dval' 'tval'};
n_classes2 = length(unique(clabel2));

if ~has_second_dataset
    if  (n_classes > 2) && any(ismember(only_binary, cfg.metric))
        error('The following metrics work only for 2 classes: %s', strjoin(only_binary))
    end
else
    need_more_than_1_class = {'auc' 'confusion' 'f1' 'precision' 'recall' 'tval'};
    if n_classes2 == 1 && any(ismember(need_more_than_1_class, cfg.metric))
        %error('Dataset 2 has only 1 class but the following metrics require more than 1 class: %s', strjoin(need_more_than_1_class))
    elseif  (n_classes2 > 2) && any(ismember(only_binary, cfg.metric))
        error('Dataset 2 has %d classes but the following metrics work only for 2 classes: %s', n_classes2, strjoin(only_binary))
    end
end

%% cfg: check for parameter names that have been changed
changed_fields = { 'nb'      'neighbours';
                      };
for ii=1:size(changed_fields, 1)
    if isfield(cfg,changed_fields{ii,1})
        cfg.(changed_fields{ii,2}) = cfg.(changed_fields{ii,1});
        cfg = rmfield(cfg,changed_fields{ii,1});
        warning('cfg.%s is now called cfg.%s, renaming argument', changed_fields{ii,1},changed_fields{ii,2})
    end
end

if any(ismember(fn, {'balance', 'replace', 'normalise'}))
    error('The fieldnames .balance / .replace / .normalise do not exist any more. Over/undersampling is now performed using the preprocessing options ''undersample'', ''oversample'', ''demean'', ''zscore''. See example7_preprocessing.m for example code.')
end

%% cfg: translate feedback specified as 'yes' or 'no' into boolean
if ischar(cfg.feedback)
    if strcmp(cfg.feedback, 'yes'),     cfg.feedback = 1;
    elseif strcmp(cfg.feedback, 'no'),  cfg.feedback = 0;
    end
end

%% cfg.preprocess: set to empty array if does not exist, and turn into cell array if it isn't yet
if ~isfield(cfg,'preprocess')
    cfg.preprocess = {};
elseif ~iscell(cfg.preprocess)
    cfg.preprocess = {cfg.preprocess};
end

if ~isfield(cfg,'preprocess_param') || isempty(cfg.preprocess_param)
    cfg.preprocess_param = {};
elseif ~iscell(cfg.preprocess_param)
    cfg.preprocess_param = {cfg.preprocess_param};
elseif iscell(cfg.preprocess_param) && ischar(cfg.preprocess_param{1})
    % in this case a cell array with key-value pairs has been passed as
    % options for the first preprocess operation, so we also wrap it
    cfg.preprocess_param = {cfg.preprocess_param};
end

%% cfg.preprocess_param: if it has less elements than .preprocess, add empty structs
if numel(cfg.preprocess_param) < numel(cfg.preprocess)
    cfg.preprocess_param(numel(cfg.preprocess_param)+1:numel(cfg.preprocess)) = {struct()};
end

%% cfg.preprocess_param: fill structs up with default parameters
for ii=1:numel(cfg.preprocess_param)
    if ischar(cfg.preprocess{ii})
        cfg.preprocess_param{ii} = mv_get_preprocess_param(cfg.preprocess{ii}, cfg.preprocess_param{ii});
    end
end

%% cfg.preprocess: add preprocessing function handles
cfg.preprocess_fun = cell(numel(cfg.preprocess), 1);
for ii=1:numel(cfg.preprocess)
    if ~isa(cfg.preprocess{ii}, 'function_handle')
        cfg.preprocess_fun{ii} = eval(['@mv_preprocess_' cfg.preprocess{ii}]);
    else
        cfg.preprocess_fun{ii} = cfg.preprocess{ii};
    end
end

%% cfg.preprocess: raise error if number of arguments in preprocess and preprocess_param does not match
if numel(cfg.preprocess) ~= numel(cfg.preprocess_param)
    error('The number of elements in cfg.preprocess and cfg.preprocess_param does not match')
end

%% cfg: set defaults for cross-validation
mv_set_default(cfg,'cv','kfold');
mv_set_default(cfg,'k',5);
mv_set_default(cfg,'p',0.1);
mv_set_default(cfg,'stratify',1);
mv_set_default(cfg,'fold',[]);

switch(cfg.cv)
    case 'leaveout' 
        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p, cfg.fold, cfg.preprocess, cfg.preprocess_param);
        cfg.k = CV.NumTestSets;
        mv_set_default(cfg,'repeat', 1);
    case 'holdout'
        cfg.k = 1;
    case 'predefined'
        if isempty(cfg.fold)
            error('if cfg.cv=''predefined'' then cfg.fold must be specified')
        end
        cfg.k = max(cfg.fold);
        mv_set_default(cfg,'repeat', 1);
end
mv_set_default(cfg,'repeat', 5);

%% X and clabel: check whether the number of instances matches the number of class labels
if numel(clabel) ~= size(X,1)
    error('Number of class labels (%d) does not match number of instances (%d) in data', numel(clabel), size(X,1))
end
if has_second_dataset && (numel(clabel2) ~= size(X2,1))
    error('Number of class labels (%d) does not match number of instances (%d) in dataset 2', numel(clabel2), size(X2,1))
end

%% check whether train and test functions are available for the classifier
if isempty(which(['train_' cfg.classifier]))
    error('Classifier %s not found: there is no train function called train_%s', cfg.classifier, cfg.classifier)
end
if isempty(which(['test_' cfg.classifier]))
    error('Classifier %s not found: there is no test function called test_%s', cfg.classifier, cfg.classifier)
end

%% libsvm: if cfg.classifier = 'libsvm', check whether it's available
if strcmp(cfg.classifier, 'libsvm')
    % We must perform sanity checks for multiple cases failure cases here:
    % (1) no svmtrain function available
    % (2) an svmtrain function is available, but it is the one by Matlab
    % (3) two svmtrain function are available (Matlab's one and libsvm's one)
    %     but the libsvm one is overshadowed by Matlab's one
    check = which('svmtrain','-all');
    msg = ' Did you install LIBSVM and add its Matlab folder to your path? Type "which(''svmtrain'',''-all'')" to check for the availability of svmtrain().';
    if isempty(check)
        error(['LIBSVM''s svmtrain() is not available or not in the path.' msg])
    else
        try
            % this should work fine with libsvm but crash for Matlab's 
            % svmtrain
            svmtrain(0,0,'-q');
        catch
            if numel(check)==1
                % there is an svmtrain but it seems to be Matlab's one
                error(['Found an svmtrain() function but it does not seem to be LIBSVM''s one.' msg])
            else
                % there is multiple svmtrain functions
                error(['Found multiple functions called svmtrain: LIBSVM''s svmtrain() is either not available or overshadowed by another svmtrain function.' msg])
            end
        end
    end
end


%% liblinear: if cfg.classifier = 'liblinear', check whether it's available
if strcmp(cfg.classifier, 'liblinear')
    % The Matlab version of liblinear uses a function called train() for
    % training. Matlab's nnet toolbox has a function of the same name.
    % We must perform sanity checks for multiple cases failure cases here:
    % (1) no train() function available
    % (2) a train() function is available, but it is the one by Matlab
    % (3) multiple train() function are available (Matlab's one and 
    %     liblinear's one but the latter one is overshadowed by Matlab's
    %     one)
    check = which('train','-all');
    msg = ' Did you install LIBLINEAR and add its Matlab folder to your path? Type "which(''train'',''-all'')" to check for the availability of train()."';
    if isempty(check)
        error(['LIBLINEAR''s train() is not available or not in the path.' msg])
    else
        try
            % this should work fine with liblinear but crash for Matlab's
            % train
            train(0,sparse(0),'-q');
        catch
            if numel(check)==1
                % there is an train but it seems to be Matlab's one
                error(['Found a train() function but it does not seem to be LIBLINEAR''s one.' msg])
            else
                % there is multiple svmtrain functions
                error(['Found multiple functions called train: LIBLINEAR''s svmtrain() is either not available or overshadowed by another svmtrain function.' msg])
            end
        end
    end
end

%% Deprecation checks
if isfield(cfg,'param') && (~isfield(cfg,'hyperparameter') || isempty(cfg.hyperparameter))
    warning('cfg.param has been renamed to cfg.hyperparameter, changing cfg accordingly..');
    cfg.hyperparameter = cfg.param;
    cfg = rmfield(cfg,'param');
end

%% cfg.hyperparameter: change default values when output_type = 'prob'
if strcmp(cfg.output_type,'prob') 
    if strcmp(cfg.classifier,'lda')
        param = cfg.hyperparameter;
        mv_set_default(param,'prob',1);
        cfg.hyperparameter = param;
    end
end

%% cfg: set defaults for classifier hyperparameter
cfg.hyperparameter = mv_get_hyperparameter(cfg.classifier, cfg.hyperparameter);
