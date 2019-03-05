function [clabel, nclasses] = mv_check_inputs(cfg, X, clabel)
% Performs some sanity checks on input parameters cfg, X, and y

%% clabel: check class labels
clabel = clabel(:);
nclasses = max(clabel);

if ~all(ismember(clabel,1:nclasses))
    error('Class labels must consist of integers 1 (class 1), 2 (class 2), 3 (class 3) and so on')
end

if numel(unique(clabel))==1
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

%% X and clabel: check whether the number of instances matches the number of class labels
if numel(clabel) ~= size(X,1)
    error('Number of class labels (%d) does not match number of instances (%d) in data', numel(clabel), size(X,1))
end

