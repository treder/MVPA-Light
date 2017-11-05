function stat = mv_statistics(cfg, result)
% Performs single-subject (level 1) or group (level 2) statistical analysis
% provided the output struct 
%
% Usage:
% stat = mv_statistics(cfg, result, <result2, ...>)
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
% cfg          - struct with parameters:
% .test        - 'binomial','ttest'
%
%
%Output:
% stat - struct with statistical output


% (c) Matthias Treder 2017







%%% --- Helper functions --- 
function print_statistics_info


end

end