function mv_check_cfg(cfg)
% Performs some sanity checks on cfg 

%% Check whether all parameters are written in lowercase

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

%% check whether different metrics are compatible with each other 
% eg 'confusion' does not work with 'tval' because the former requires
% class labels as ouput whereas the latter requires dvals
incompatible_metrics = { 'confusion' {'auc' 'tval' 'dval'};
    };

idx = find(ismember(incompatible_metrics(:,1), cfg.metric));

if any(idx) && any(ismember(incompatible_metrics{idx,2}, cfg.metric))
    error('The metric ''%s'' cannot be calculated together with metrics %s', incompatible_metrics{idx,1}, strjoin(incompatible_metrics{idx,2}))
end

%% check whether classifier and metric are compatible (eg 'auc' does not work for multiclass_lda)

% Combinations of classifier and metrics that do not work together
classifier_metric = { 'multiclass_lda' {'auc' 'tval' 'dval'};
                      'kernel_fda'     {'auc' 'tval' 'dval'};
    };

idx = find(ismember(classifier_metric(:,1), cfg.classifier));
if any(idx) && any(ismember(classifier_metric{idx,2}, cfg.metric))
    error('The following metrics cannot be used with %s: %s', cfg.classifier, strjoin(classifier_metric{idx,2}))
end
