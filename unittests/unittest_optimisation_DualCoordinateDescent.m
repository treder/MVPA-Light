function unittest_optimisation_DualCoordinateDescent
% Test of optimisation functions
%
% Function: DualCoordinateDescent
%
% The function solves quadratic optimisation problems with box constraints,
% more concretely:
%
%    arg min a     f(a) = 1/2 a' * Q * a - e' * a
%    subject to    0 <= a <= C
%
% A few such problems will be tested here. 

nfeat = 20;
ONE = ones(nfeat,1);
C = 5;

%% Q = diagonal with high cost and low cost directions
% Create diagonal
d = ones(nfeat,1);
d(1:floor(nfeat/2)) = 1/1000;       % "low cost" entries 
d(floor(nfeat/2)+1:end) = -10;         % "high cost" entries

% Transform into matrix
Q = diag(d);

tol = 10e-10;
[alpha,iter] = DualCoordinateDescent(Q,C,ONE, tol, 1);

% low cost entries should be high values of alpha and vice versa
print_unittest_result('Q=diagonal (lo cost directions = C)', C, mean(alpha(1:floor(nfeat/2))), tol);
print_unittest_result('Q=diagonal (hi cost directions = 0)', 0, mean(alpha(floor(nfeat/2)+1:end)), tol);

%% Q = 0
Q = zeros(nfeat); eye(nfeat);

% All alpha's should be at maximum (since there's no quadratic penalty)
alpha = DualCoordinateDescent(Q,C,ONE, tol, 1);

print_unittest_result('Q=0 (all alpha=C?)', C, min(alpha), tol);

%% Q = 0
Q = eye(nfeat);

% the first two rows/column are 1 = penalised most
Q(:,1:2) = 1;
Q(1:2,:) = 1;

[alpha, iter] = DualCoordinateDescent(Q,C,ONE, tol, 1);

% Alphas for the first two rows/columns should be zero
print_unittest_result('Q=0 (for alpha(1:2))', 0, max(alpha(1:2)), tol);

