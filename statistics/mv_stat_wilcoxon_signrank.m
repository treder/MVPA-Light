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

sz = size(X);

% Alternative: just call matlab's signrank function on each column (but
% this is much slower)
% zval = zeros([1, sz(2:end)]);
% for ix = 1:prod(sz(2:end))
%     [~,~,stat] = signrank(X(:,ix), 0, 'method', 'approximate');
%     zval(ix) = stat.zval;
% end
% return

% remove 0 values
X(X==0) = nan;

% sort according to absolute value
X = sort(X, 1, 'ComparisonMethod', 'abs');

ranks = repmat((1:sz(1))', [1, sz(2:end)]);

% duplicate values need special treatment. To avoid using the unique()
% function (because slow) we can calculate the diff along the first
% dimension and then look for 0's
is_duplicate = (diff(abs(X),1,1) == 0);

% there can be multiple chunks of duplicates. Start of a chunk can be 
% recognized as diff(is_duplicate) == 1. End of a chunk corresponds to
% diff(is_duplicate) == -1
diff_duplicate = diff(is_duplicate);
chunk_start = (diff_duplicate == 1);
chunk_end = (diff_duplicate == -1);
% the above does not recognize chunk starts at position 1 or chunk ends at 
% the last position, so we add these separately as an extra row
chunk_start = cat(1, reshape(is_duplicate(1,:), [1, sz(2:end)]), chunk_start);
chunk_end = cat(1, zeros([1, sz(2:end)]), chunk_end, reshape(is_duplicate(end,:), [1, sz(2:end)]));
n_chunks = sum(chunk_start,1);

% check for duplicate values in each column and update ranks of duplicates
for ix = 1:prod(sz(2:end))
    if n_chunks(ix) > 0
        % there can be multiple chunks of duplicates
        st = find(chunk_start(:,ix));
        en = find(chunk_end(:,ix));
        for ii = 1:n_chunks(ix)
            ranks(st(ii):en(ii), ix) = mean(ranks(st(ii):en(ii), ix));            
        end
    end
end

% count number of non-nan values per column
Nr = sum(~isnan(X),1); 

% for calculating the test statistic replace the Nan back to 0's
ranks(isnan(X)) = 0;
X(isnan(X)) = 0;

% calculate test statistic
w = sum(sign(X) .* ranks, 1);

% calculate z-value
sigma = sqrt(Nr .* (Nr + 1) .* (2*Nr + 1) / 6);
zval = w ./ sigma;
    
if nargout > 1
    % approximate two-tailed p-value using normal distribution
    p = 2 * (1- normcdf(zval, 0, 1));
end

