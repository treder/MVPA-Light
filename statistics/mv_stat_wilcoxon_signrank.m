function [zval, p, w] = mv_stat_wilcoxon_signrank(X, X2)
% Two-tailed Wilcoxon signed rank test (one sample or two samples). 
% Unlike Matlab's signrank function, it also works for multi-dimensional
% arrays.
%
% Usage:
% [stat, p] = mv_stat_wilcoxon_signrank(X, <X2>)
%
%Parameters:
% X              - [samples x ... x ...] matrix of samples that were used
%                  for training the classifier
% X2             - (optional) matrix of second sample for two-samples test.
%
% If X2 is not provided, a one-sample test against the hypothesis
% median = 0 is performed.
%
%Returns:
% zval           - z-value given as the normalized statistic w / std(w)
% p              - two-tailed p-value. A normal approximation is used but
%                  it is only a good approximation for >20 samples.
% w              - signed rank statistic
%
%Reference:
% https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
%
% (c) Matthias Treder 2020

if nargin==2
    X = X - X2;
end

if isvector(X), X = X(:); end

SGN = sign(X);
X = abs(X);
sz = size(X);

% remove 0 values
X(X==0) = nan;

% rank according to absolute value
[~, ranks] = sort(X, 1);

% check for duplicate values in each column and update ranks of duplicates
for ix = 1:prod(sz(2:end))
    % transform into ranks
    ranks(ranks(:,ix), ix) = 1:sz(1);
    [~,ind] = unique(X(:,ix));
    if numel(ind) < sz(1)
        for ix_nonunique = setdiff(1:sz(1), ind)
            % find all indices for the non-unique element
            sel = (X(:, ix) == X(ix_nonunique, ix));
            % set rank to average of the ranks they occupy
            ranks(sel, ix) = mean(ranks(sel, ix));
        end 
    end
end

% for calculating the test statistic replace the Nan back to 0's
ranks(isnan(X)) = 0;

% calculate test statistic
w = sum(SGN .* ranks, 1);

% calculate z-value
Nr = sum(abs(SGN), 1); % this will count the number of non-nan values
sigma = sqrt(Nr .* (Nr + 1) .* (2*Nr + 1) / 6);
zval = w ./ sigma;
    
if nargout > 1
    % calculate approximate p-value using normal approximation
    p = 2 * (1- normcdf(zval, 0, 1));
end

