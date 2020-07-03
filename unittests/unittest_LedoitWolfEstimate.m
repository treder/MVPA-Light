rng(42)
tol = 10e-10;

N = 1000;
P = 100;

X = simulate_gaussian_data(N, P, 1, [], [], 0);
X2 = randn(N, P);
X3 = simulate_spiral_data(N, 3, 1);

LW = @LedoitWolfEstimate;  % short notation

%% result between 0 and 1
print_unittest_result('[Gaussian data primal] result between 0 and 1', true, 0 <= LW(X, 'primal') <= 1, tol);
print_unittest_result('[randn data primal] result between 0 and 1', true, 0 <= LW(X2, 'primal') <= 1, tol);
print_unittest_result('[spiral data primal] result between 0 and 1', true, 0 <= LW(X3, 'primal') <= 1, tol);

print_unittest_result('[Gaussian data dual] result between 0 and 1', true, 0 <= LW(X, 'dual') <= 1, tol);
print_unittest_result('[randn data dual] result between 0 and 1', true, 0 <= LW(X2, 'dual') <= 1, tol);
print_unittest_result('[spiral data dual] result between 0 and 1', true, 0 <= LW(X3, 'dual') <= 1, tol);

%% compare with cov1para [external function written by Ledoit and Wolf]
[~,lambda]=cov1para(X);
print_unittest_result('[Gaussian] compare with cov1para', LW(X, 'primal'), lambda, tol);
[~,lambda]=cov1para(X2);
print_unittest_result('[randn data] compare with cov1para', LW(X2, 'primal'), lambda, tol);
[~,lambda]=cov1para(X3);
print_unittest_result('[spiral data] compare with cov1para', LW(X3, 'primal'), lambda, tol);

%% equal result for primal vs dual approach?
print_unittest_result('[Gaussian] compare primal and dual result', LW(X, 'primal'), LW(X, 'dual'), tol);
print_unittest_result('[randn data] compare primal and dual result', LW(X2, 'primal'), LW(X2, 'dual'), tol);
print_unittest_result('[spiral data] compare primal and dual result', LW(X3, 'primal'), LW(X3, 'dual'), tol);

%% also test P >> N and N >> P data
X = rand(10, 1000);  % P >> N
X2 = rand(1000, 10);  % N >> P
print_unittest_result('[P >> N data] compare primal and dual result', LW(X, 'primal'), LW(X, 'dual'), tol);
print_unittest_result('[N >> P data] compare primal and dual result', LW(X2, 'primal'), LW(X2, 'dual'), tol);

[~,lambda]=cov1para(X);
print_unittest_result('[P >> N data] compare with cov1para.m', LW(X, 'primal'), lambda, tol);
[~,lambda]=cov1para(X2);
print_unittest_result('[N >> P data] compare with cov1para.m', LW(X2, 'primal'), lambda, tol);
