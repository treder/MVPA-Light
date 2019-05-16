function [cfg, clabel, nclasses] = mv_check_inputs(cfg, X, clabel)
% Performs some sanity checks on input parameters cfg, X, and y.
% Also checks whether external toolboxes (LIBSVM and LIBLINEAR) are
% available if required.

if ~iscell(cfg.metric)
    cfg.metric = {cfg.metric};
end

%% clabel: check class labels
clabel = clabel(:);
u = unique(clabel);
nclasses = length(u);

if ~all(ismember(clabel,1:nclasses))
    warning('Class labels should consist of integers 1 (class 1), 2 (class 2), 3 (class 3) and so on. Relabelling them accordingly.');
    newlabel = nan(numel(clabel), 1);
    for i = 1:nclasses
        newlabel(clabel==u(i)) = i; % set to 1:nth classes
    end
    clabel = newlabel;
end

if nclasses==1
    error('Only one class specified. Class labels must contain at least 2 classes')
end

%% clabel and cfg: check whether there's more than 2 classes but yet a binary classifier is used
binary_classifiers = {'lda' 'logreg' 'svm'};
if nclasses > 2 && ismember(cfg.classifier, binary_classifiers)
    error('Cannot use %s for a classification task with %d classes: use a multiclass classifier instead', upper(cfg.classifier), nclasses)
end

%% cfg: check whether all parameters are written in lowercase
fn = fieldnames(cfg);

% Are all cfg fields given in lowercase?
not_lowercase = find(~strcmp(fn,lower(fn)));

if any(not_lowercase)
    error('For consistency, all parameters must be given in lowercase: please replace cfg.%s by cfg.%s', fn{not_lowercase(1)},lower(fn{not_lowercase(1)}) )
end

% Are all cfg.param fields given in lowercase?
if isfield(cfg,'param') && isstruct(cfg.param)
    pfn = fieldnames(cfg.param);
    not_lowercase = find(~strcmp(pfn,lower(pfn)));
    
    if any(not_lowercase)
        error('For consistency, all parameters must be given in lowercase: please replace param.%s by param.%s', pfn{not_lowercase(1)},lower(pfn{not_lowercase(1)}) )
    end
end

%% cfg: check whether different metrics are compatible with each other 
% eg 'confusion' does not work with 'tval' because the former requires
% class labels as ouput whereas the latter requires dvals
incompatible_metrics = { 'confusion' {'auc' 'tval' 'dval'};
    };

idx = find(ismember(incompatible_metrics(:,1), cfg.metric));

if any(idx) && any(ismember(incompatible_metrics{idx,2}, cfg.metric))
    error('The metric ''%s'' cannot be calculated together with metrics %s', incompatible_metrics{idx,1}, strjoin(incompatible_metrics{idx,2}))
end

%% cfg: check whether classifier and metric are compatible (eg 'auc' does not work for multiclass_lda)

% Combinations of classifier and metrics that do not work together
classifier_metric = { 'multiclass_lda' {'auc' 'tval' 'dval'};
                      'kernel_fda'     {'auc' 'tval' 'dval'};
    };

idx = find(ismember(classifier_metric(:,1), cfg.classifier));
if any(idx) && any(ismember(classifier_metric{idx,2}, cfg.metric))
    error('The following metrics cannot be used with %s: %s', cfg.classifier, strjoin(classifier_metric{idx,2}))
end

%% cfg: check whether metrics are compatible with output_type
% eg 'confusion' does not work with 'tval' because the former requires
% class labels as ouput whereas the latter requires dvals
metric_outputtype = { 'auc'       {'dval' 'prob'};
                      'dval'      {'dval' 'prob'};
                      'tval'      {'dval' 'prob'};
                      'confusion' 'clabel';
                      'precision' 'clabel';
                      'recall'    'clabel';
                      'f1'        'clabel'
                      };

idx = find(ismember(metric_outputtype(:,1), cfg.metric));

for ii=1:numel(idx)
    if ~any(strcmp(metric_outputtype{idx(ii),2}, cfg.output_type))
        error('The metric ''%s'' requires %s as output_type', metric_outputtype{idx(ii),1},metric_outputtype{idx(ii),2})
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

%% cfg: translate feedback specified as 'yes' or 'no' into boolean
if ischar(cfg.feedback)
    if strcmp(cfg.feedback, 'yes'),     cfg.feedback = 1;
    elseif strcmp(cfg.feedback, 'no'),  cfg.feedback = 0;
    end
end

%% X and clabel: check whether the number of instances matches the number of class labels
if numel(clabel) ~= size(X,1)
    error('Number of class labels (%d) does not match number of instances (%d) in data', numel(clabel), size(X,1))
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
            train(0,0,'-q');
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
