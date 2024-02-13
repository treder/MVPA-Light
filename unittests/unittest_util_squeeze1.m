% util unit test
%
% squeeze1
tol = 10e-10;

%% special case that X is scalar
X = 12.2;
print_unittest_result('[X scalar]', size(X), size(squeeze1(X)), tol);

%% squeeze = squeeze1 for 2 elements
X = ones(13, 16);
print_unittest_result('[2 dims] no singleton', [13, 16], size(squeeze1(X)), tol);
X = ones(80, 1);
print_unittest_result('[2 dims] 2nd element singleton', [80, 1], size(squeeze1(X)), tol);
X = ones(1, 80);
print_unittest_result('[2 dims] 1st element singleton', [1, 80], size(squeeze1(X)), tol);

%% squeeze = squeeze1 for >=3 elements
X = ones(10, 1, 20);
print_unittest_result('[3 dims] middle element singleton', size(squeeze(X)), size(squeeze1(X)), tol);
X = ones(10, 14, 20);
print_unittest_result('[3 dims] no singletons', size(squeeze(X)), size(squeeze1(X)), tol);
X = ones(10, 1, 10, 20, 5);
print_unittest_result('[5 dims] 2nd element singleton', size(squeeze(X)), size(squeeze1(X)), tol);
X = ones(10, 1, 10, 1, 5);
print_unittest_result('[5 dims] dim 2, 4 singleton', size(squeeze(X)), size(squeeze1(X)), tol);
X = ones(10, 1, 1, 1, 5);
print_unittest_result('[5 dims] dim 2, 3, 4 singleton', size(squeeze(X)), size(squeeze1(X)), tol);
X = ones(1, 1, 1, 1, 5);
print_unittest_result('[5 dims] dim 1-4 singleton', [1 5], size(squeeze1(X)), tol);
X = ones(1, 1, 5, 1, 1);
print_unittest_result('[5 dims] dim 1, 2, 4, 5 singleton', [1 5], size(squeeze1(X)), tol);
X = ones(1, 5, 1, 1, 1);
print_unittest_result('[5 dims] dim 1, 3, 4, 5 singleton', [1 5], size(squeeze1(X)), tol);
