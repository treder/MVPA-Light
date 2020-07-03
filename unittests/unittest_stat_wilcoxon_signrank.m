% unit test of mv_stat_wilcoxon_signrank

rng(42)
tol = 10e-10;

% example X with duplicates for debugging:
% X = [10 -4 ; 1 10; 40 4; 10, -20; 0 -2; -30 20; 40 2; -10 10]

%% [X is a vector] check whether yields the same p-value as matlab's signrank
X = randn(100,1)*10 + 2;

p1 = signrank(X, 0, 'method','approximate');
[~,p2] = mv_stat_wilcoxon_signrank(X);

print_unittest_result('[X is vector] compare p-value to Matlab''s signrank function', p1, p2, tol);

%% [X is a matrix] check whether yields the same p-value as matlab's signrank
X = randn(200,5,3)*10 + 2;

p1 = [];
sz = size(X);
% run signrank for each column in X
for ix = 1:prod(sz(2:end))
    p1 = [p1, signrank(X(:,ix), 0, 'method','approximate')];
end
p1 = reshape(p1, [1, sz(2:end)]);

[~,p2] = mv_stat_wilcoxon_signrank(X);

print_unittest_result('[X is multidimensional] compare p-values to Matlab''s signrank function', p1, p2, tol);

%% check multi-dimensional data
X = randn(100,3)*10 + 2;

% copy X along 3rd dimension. Check whether the results are consistent for
% X(:,:,1) and X(:,:,2)
X = repmat(X, 1, 1, 2);

[~,p] = mv_stat_wilcoxon_signrank(X);

print_unittest_result('consistent results for multidim data', p(:,:,1), p(:,:,2), tol);

%% two-samples Wilcoxon test
% (using example dataset from Wikipedia page)

X = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135];
Y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145];

[~,p, w] = mv_stat_wilcoxon_signrank(X, Y);

print_unittest_result('[two-samples test] w statistic', 9, w, tol);
print_unittest_result('[two-samples test] p value', 0.6113, p, 10e-1); % wikipedia uses the exact p-value we only have an approximation
