function [zval, p, U] = mv_stat_wilcoxon_ranksum(X, group, n1, n2)
% Two-tailed Wilcoxon rank sum test (equivalent to the Mann-Whitney U test).
% Unlike Matlab's ranksum function, it also works for multi-dimensional
% arrays.
%
% Usage:
% [zval, p, U] = mv_stat_wilcoxon_ranksum(X, group, <n1, n2>)
% [stat, p] = mv_stat_wilcoxon_ranksum(X1, X2)
%
%Parameters:
% X              - [samples x ... x ...] matrix of data from two samples
% group          - [samples x 1] vector of 1's and 2's signifying which row
%                   of X belongs to which group
% n1, n2         - (optional) number of samples in each group
%
%Returns:
% zval           - z-value given as the normalized statistic w / std(w)
% p              - two-tailed p-value. A normal approximation is used but
%                  it is only a good approximation for >20 samples.
% U              - Mann-Whitney U-statistic. It is related to the Wilcoxon 
%                  ranksum statistic W as U = W - n1*(n1+1)/2
%
%Reference:
% https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
% https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Area-under-curve_(AUC)_statistic_for_ROC_curves

% (c) Matthias Treder

if isvector(X), X = X(:); end
if nargin<3, n1 = sum(group==1); end
if nargin<4, n2 = sum(group==2); end

% calculate test statistic U. It is closely related to AUC:
% https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Area-under-curve_(AUC)_statistic_for_ROC_curves
U = mv_calculate_performance('auc','dval', X, group) * n1 * n2;
U = min(U, n1*n2 - U); % whichever is smaller

% calculate z-value using a normal approximation
% https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Normal_approximation_and_tie_correction
% Note: tied ranks are ignored since the data is generally continuous and ties are not very frequent
mU = (n1*n2)/2;
sigma = sqrt( n1*n2*(n1+n2+1)/12 ); 
zval = -(U - mU) ./ sigma;
    
if nargout > 1
    % approximate two-tailed p-value using normal distribution
    p = 2 * (1 - normcdf(zval, 0, 1));
end

