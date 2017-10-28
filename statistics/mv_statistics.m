function stat = mv_statistics(cfg, varargin)
% Usage:
% stat = mv_statistics(cfg, result1, <out2, ...>)
%
%Parameters:
% out1         - struct describing the classification outcome. Can be
%                obtained as second output argument from functions
%                mv_crossvalidate, mv_classify_across_time,
%                mv_classify_timextime, and mv_searchlight. 
%                For group analysis (across subjects), a struct array should
%                be provided where each element corresponds to one subject.
%                For instance, out1(1) corresponds to the first subject,
%                out1(2) to the second, and so on.
% 
%                In case of multiple conditions, additional structs or
%                struct arrays can be provided as additional input arguments 
%                out2, out3, etc.
%
% cfg          - struct with parameters:
% .test        - 'binomial'
%
% .prob          - if 1, probabilities are returned as decision values. If
%                  0, the decision values are simply the distance to the
%                  hyperplane. Calculating probabilities takes more time
%                  and memory so don't use this unless needed (default 0)
% .scale         - if 1, the projection vector w is scaled such that the
%                  mean of class 1 (on the training data) projects onto +1
%                  and the mean of class 2 (on the training data) projects
%                  onto -1
%
%Output:
% stat - struct with statistical output


% (c) Matthias Treder 2017


