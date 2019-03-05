function stat = mv_statistics(cfg, result)
% Performs single-subject (level 1) or group (level 2) statistical analysis
% on the classifier performance measures.
%
% Usage:
% stat = mv_statistics(cfg, result)
%
%Parameters:
% result       - struct describing the classification outcome. Can be
%                obtained as second output argument from functions
%                mv_crossvalidate, mv_classify_across_time,
%                mv_classify_timextime, and mv_searchlight.
%
%                For group analysis (across subjects), a cell array should
%                be provided where each element corresponds to one subject.
%                For instance, result{1} corresponds to the first subject,
%                result{2} to the second, and so on.
% 
%                In case of multiple conditions, additional structs or
%                struct arrays can be provided as additional input arguments 
%                out2, out3, etc.
%
% cfg is a struct with parameters:
% .test        - specify the statistical test that is applied. Some tests
%                are applied to a single subject (and hence need only one
%                result struct as input), some are applied across subjects
%                to a group and hence need a cell array as input
%                'binomial': binomial test [single-subject analysis]
%                'permutation'
% .alpha       - significance threshold (default 0.05)
% .chance      - chance level (default 0.5)
%
% Further details regarding specific tests:
% BINOMIAL TEST (single-subject analysis):
% Uses a binomial distribution ...
%%% NOTE: we can't add the repeats as separate samples, since the samples
%%% are the same = independence assumption violated. So we always take
%%% number of samples, ignoring the number of repeats
% But how do we treat results from a cross-validation analysis: since the 
% sum (which is the unnormalised mean) of binomially distributed variables 
% is binomial, too, we can treat the results on the folds and repetitions
% as a single large binomial test. This is possible because the 
% classification accuracy has been calculated using weighted averaging, and
% hence the total number of hits is equal to the average accuracy *  total
% number of samples.
%
%Output:
% stat - struct with statistical output

% (c) Matthias Treder 2017

mv_set_default(cfg,'alpha', 0.05);
mv_set_default(cfg,'chance', 0.5);
mv_set_default(cfg,'feedback', 1);

%% Statistical testing
stat = struct('test',cfg.test,'statistic',[],'p',[]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     BINOMIAL TEST     %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(cfg.test,'binomial')

    % N is the total number of samples
    n = result.n;
    
    % Calculate p-value using the cumulative distribution function, testing
    % H0: the observed accuracy was due to chance
    stat.p = 1 - binocdf( round(result.perf * n), n, cfg.chance);

elseif strcmp(cfg.test,'permutation')
end

stat.mask = stat.p < cfg.alpha;


%% Print output
if cfg.feedback
    fprintf('\nPerforming a %s test\n',upper(cfg.test))
    fprintf('p-value(s): %s\n',sprintf('%0.3f ',stat.p) )
    fprintf('significant (p > alpha): %s\n',sprintf('%d ',stat.mask) )
end


end