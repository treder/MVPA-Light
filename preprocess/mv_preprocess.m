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
% .preprocess_fun     - cell array containing function handles to the
%                       preprocessing functions. cfg.preprocess_fun is
%                       generated from cfg.preprocessin mv_classify and
%                       other high-level functions
%                       pipeline is applied in chronological order
% .preprocess_param   - cell array of preprocessing parameter structs for each
%                       function. Length of preprocess_param must match length
%                       of preprocess

for pp=1:numel(cfg.preprocess_fun)   % -- loop over preprocessing pipeline
    
    % call preprocessing function
    [cfg.preprocess_param{pp}, X, clabel] = cfg.preprocess_fun{pp}(cfg.preprocess_param{pp}, X, clabel);
    
    % the current preprocessing step may affect downstream preprocessing, e.g.
    % when undersampling and subsequent confounding class specific mean subtraction
    if cfg.preprocess_param{pp}.is_train_set && isfield(cfg.preprocess_param{pp}, 'keep_idx_train')
      for p = (pp+1):numel(cfg.preprocess_fun)
        for fn = cfg.preprocess_param{p}.select_data
          cfg.preprocess_param{p}.([fn{1} '_train']) = cfg.preprocess_param{p}.([fn{1} '_train'])(cfg.preprocess_param{pp}.keep_idx_train,:,:,:);
        end
      end
    elseif ~cfg.preprocess_param{pp}.is_train_set && isfield(cfg.preprocess_param{pp}, 'keep_idx_test')
      for p = (pp+1):numel(cfg.preprocess_fun)
        for fn = cfg.preprocess_param{p}.select_data
          cfg.preprocess_param{p}.([fn{1} '_test'])  = cfg.preprocess_param{p}.([fn{1} '_test'])(cfg.preprocess_param{pp}.keep_idx_test,:,:,:);
        end
      end
    end
    
    % swap between train/test set
    cfg.preprocess_param{pp}.is_train_set = 1 - cfg.preprocess_param{pp}.is_train_set;

end