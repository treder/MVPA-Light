function [cfg, Y, nmetrics] = mv_check_inputs_for_regression(cfg, X, Y)
% For regression: Performs some sanity checks and sets some defaults for 
% input parameters cfg, X, and y.
% Also checks whether external toolboxes (LIBSVM and LIBLINEAR) are
% available if required.

if ~iscell(cfg.metric)
    cfg.metric = {cfg.metric};
end
nmetrics = numel(cfg.metric);

%% Y: check whether it is a column vector
if isvector(Y) && numel(Y)>1 && size(Y,2) > 1
    error('Response vector Y should be given as a column vector, not a row vector')
end

%% Y: if Y is multivariate, check whether this is supported by the model

if ~isvector(Y)
    % todo
end

%% X and Y: check whether the number of instances is matched
if isvector(Y), len = length(Y);
else, len = size(Y,1); end
if len ~= size(X,cfg.sample_dimension)
    error('Number of responses (%d) does not match number of instances (%d) in data', len, size(X,cfg.sample_dimension))
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

%% cfg: set defaults for cross-validation
mv_set_default(cfg,'cv','kfold');
mv_set_default(cfg,'repeat',5);
mv_set_default(cfg,'k',5);
mv_set_default(cfg,'p',0.1);
mv_set_default(cfg,'fold',[]);

switch(cfg.cv)
    case 'leaveout', cfg.k = size(X, cfg.sample_dimension);
    case 'holdout', cfg.k = 1;
end

%% cfg: set defaults for model hyperparameter
cfg.hyperparameter = mv_get_hyperparameter(cfg.model, cfg.hyperparameter);

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

%% cfg.preprocess: convert preprocessing function to function handle
for ii=1:numel(cfg.preprocess)
    if ~isa(cfg.preprocess{ii}, 'function_handle')
        cfg.preprocess{ii} = eval(['@mv_preprocess_' cfg.preprocess{ii}]);
    end
end

%% cfg.preprocess: raise error if number of arguments in preprocess and preprocess_param does not match
if numel(cfg.preprocess) ~= numel(cfg.preprocess_param)
    error('The number of elements in cfg.preprocess and cfg.preprocess_param does not match')
end

%% check whether train and test functions are available for the regression model
if isempty(which(['train_' cfg.model]))
    error('Regression model ''%s'' not found: there is no train function called train_%s', cfg.model, cfg.model)
end
if isempty(which(['test_' cfg.model]))
    error('Regression model ''%s'' not found: there is no test function called test_%s', cfg.model, cfg.model)
end

%% libsvm: if cfg.model = 'libsvm', check whether it's available
if strcmp(cfg.model, 'libsvm')
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


%% liblinear: if cfg.model = 'liblinear', check whether it's available
if strcmp(cfg.model, 'liblinear')
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

