function varargout = mv_classify_searchlight(cfg, X, label)
% Searchlight classification. Classification is performed for each feature
% (or each neighborhood of features) separately. Uses
% mv_crossvalidate to perform the classification.
% Returns classification performance for each feature.
%
%
% Usage:
% [perf, ...] = mv_classify_searchlight(cfg,X,label)
%
%Parameters:
% X              - [number of samples x number of features]
%                  data matrix or a 
%                  [samples x features x time] matrix. In the latter case,
%                  all time points are additionally used as features.
% labels         - [number of samples] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with parameters:
% .nb          - [number of features x number of features] matrix of 0's 
%                 and 1's describing the neighborhood structure. A 1 in the
%                 i-th row, j-th column indicates that feature i and j are
%                 "neighbours". 
% .step        - 
%
% See mv_crossvalidate for cross-validation and other parameters.
%
% Returns:
% perf          - [features x 1] vector of classifier performances. 

% (c) Matthias Treder 2017

mv_setDefault(cfg,'nb',[]);
mv_setDefault(cfg,'metric','auc');

