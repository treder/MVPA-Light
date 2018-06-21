function mv_check_cfg(cfg)
% Performs some sanity checks on cfg and checks whether all parameters
% are written in lowercase

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

