function preprocess_param = mv_get_preprocess_param(preprocess, preprocess_param)
% Returns a parameter struct with default parameters for a given
% preprocessing function.
%
%Usage:
% param = mv_get_preprocess_param(preprocess_fun, <param>)
% 
%Parameters:
% preprocess_fun   - [string] specifying the preprocessing function (e.g. 'average_samples')
% preprocess_param - [struct or cell array] (optional) contains parameters. 
%                    If cell array, will be converted to struct. The struct
%                    will be filled up with default values for non-existing
%                    fields, but existing values are not overwritten. If
%                    param is not provided, all values are set to default
%
% Alternatively, multiple preprocessing functions and parameter sets can be
% provided as cell arrays. In this case, mv_get_preprocess_param is
% repeatedly called for each element 
%
%Output:
% param  - [struct] with default preprocessing parameters

if nargin < 2
    preprocess_param = struct();
elseif iscell(preprocess_param)
    preprocess_param = mv_cell2struct(preprocess_param);
elseif  ~isstruct(preprocess_param)
    preprocess_param = struct();
end

if iscell(preprocess)
    % multiple preprocessing function have been provided as cell array,
    % process them recursively
    for pp=1:numel(preprocess)
        preprocess_param{pp} = mv_get_preprocess_param(preprocess{pp}, preprocess_param{pp});
    end
end

switch(preprocess)
    
    case 'average_kernel'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'group_size',5);
        mv_set_default(preprocess_param,'select_data',[]);
    
    case 'average_samples'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'group_size',5);
        mv_set_default(preprocess_param,'select_data',[]);
        
    case 'csp'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'feature_dimension',2);
        mv_set_default(preprocess_param,'target_dimension',3);
        mv_set_default(preprocess_param,'lambda',10^-10);
        mv_set_default(preprocess_param,'n',3);
        mv_set_default(preprocess_param,'calculate_variance',false);
        mv_set_default(preprocess_param,'calculate_log',true);
        mv_set_default(preprocess_param,'calculate_spatial_pattern',false);
        mv_set_default(preprocess_param,'select_data',[]);

    case 'demean'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'dimension',1);
        mv_set_default(preprocess_param,'select_data',[]);

    case 'oversample'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'sample_dimension',1);
        mv_set_default(preprocess_param,'oversample_test_set',0);
        mv_set_default(preprocess_param,'replace',1);
        mv_set_default(preprocess_param,'select_data',[]);
        
    case 'pca'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'n',20);
        mv_set_default(preprocess_param,'feature_dimension',2);
        mv_set_default(preprocess_param,'target_dimension',3);
        mv_set_default(preprocess_param,'normalize',1);
        mv_set_default(preprocess_param,'select_data',[]);
    
    case 'replacenan'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'sample_dimension',1);
        mv_set_default(preprocess_param,'select_data',[]);
   
    case 'ssd'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'feature_dimension',2);
        mv_set_default(preprocess_param,'target_dimension',3);
        mv_set_default(preprocess_param,'lambda',10^-10);
        mv_set_default(preprocess_param,'n',5);
        mv_set_default(preprocess_param,'signal',[]);
        mv_set_default(preprocess_param,'noise',[]);
        mv_set_default(preprocess_param,'calculate_variance',false);
        mv_set_default(preprocess_param,'calculate_log',true);
        mv_set_default(preprocess_param,'calculate_spatial_pattern',false);
        mv_set_default(preprocess_param,'select_data', {'signal' 'noise'});

    case 'undersample'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'sample_dimension',1);
        mv_set_default(preprocess_param,'undersample_test_set',0);
        mv_set_default(preprocess_param,'select_data',[]);
        
    case 'zscore'
        mv_set_default(preprocess_param,'is_train_set',1);
        mv_set_default(preprocess_param,'dimension',1);     
        mv_set_default(preprocess_param,'select_data',[]);
end