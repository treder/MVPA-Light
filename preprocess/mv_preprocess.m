function [cfg, X, clabel] = mv_preprocess(cfg, X, clabel)
% Applies a preprocessing pipeline to the data (train or test).
%
% This function is usually called automatically from within the high-level
% functions such as mv_classify_across_time.
%
% Usage:
% [cfg, X, clabel] = mv_preprocess(cfg, X, clabel)
%
%Parameters:
% X              - [... x ... x ... x ] data matrix
% clabel         - [samples x 1] vector of class labels 
%
% cfg     - struct with preprocessing parameters:
% .preprocess         - cell array containing the preprocessing pipeline. The
%                       pipeline is applied in chronological order
% .preprocess_param   - cell array of preprocessing parameter structs for each
%                       function. Length of preprocess_param must match length
%                       of preprocess

for pp=1:numel(cfg.preprocess)   % -- loop over preprocessing pipeline
    
    % call preprocessing function
    [cfg.preprocess_param{pp}, X, clabel] = cfg.preprocess{pp}(cfg.preprocess_param{pp}, X, clabel);
    
    % swap between train/test set
    cfg.preprocess_param{pp}.is_train_set = 1 - cfg.preprocess_param{pp}.is_train_set;

end