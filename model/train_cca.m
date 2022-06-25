function model = train_cca(param, X, Y)
% Performs a regularized canonical correlation analysis (rCCA).
% Regularization is implemented as a convex combination the empirical
% covariance matrix and an identity matrix:
%
% C_reg = (1-lambda) * C_orig + lambda * trace(C_orig) * I
%
% Usage:
% cf = train_cca(param,X,clabel)
%
% Parameters:
% X              - [samples x P] matrix of training samples
% Y              - [samples x Q] matrix of training samples. 
%                  The number of features P and Q in X and Y can be
%                  different, but the number of samples N must be equal.
%
% param          - struct with hyperparameters:
% .n          - number of components (default 'auto' is equal to min(Q,P))
% .lambda_x   - regularization paramter in [0,1] for X variable (default 0
%              = no regularization). Set to 'auto' to use automatic
%              Ledoit-Wolf regularization.
% .lambda_y  - regularization paramter in [0,1] for Y variable (default 0
%              = no regularization). Set to 'auto' to use automatic
%              Ledoit-Wolf regularization.
%              We write the regularized covariance matrix as a convex combination of
% .form          - uses the 'primal' or 'dual' form of the solution to
%                  determine w. If 'auto', auomatically chooses the most 
%                  efficient form depending on whether #samples > #features
%                  (default 'auto').
% .pattern     if 1 also calculates the canonical loadings (spatial patterns) (default 1)
% .correlation if 1 also calculates the correlations between the X and Y covariates and p-values 
%
%Output:
% model - struct specifying the regression model with the following fields:
% Xw, Yw        - matrix of weight vectors for X and Y (in rows)
% Xv, Yv        - canonical variates (projections of X, Y onto Xw, Yw)
% r, p          - r and p-values for correlations between Xv and Yv
%
%Reference:
% HD Vinod. "Canonical ridge and econometrics of joint production".
% Journal of Econometrics, Volume 4, Issue 2, May 1976, Pages 147-166
%

N = size(X,1);
P = size(X,2);
Q = size(Y,2);

assert(size(X,1)==size(Y,1), 'X and Y must have the same number of samples')
if ischar(param.n) && strcmp(param.n,'auto')
    param.n = min(P, Q);
else
    assert(param.n <= min(P,Q), 'n must be smaller or equal to the number of features in X or Y')
end

model = struct();

%% Center X and Y
model.X_mean = mean(X);
model.Y_mean = mean(Y);
X = X - repmat(model.X_mean, N, 1);
Y = Y - repmat(model.Y_mean, N, 1);

%% Choose between primal and dual form
if strcmp(param.form, 'auto')
    if N >= P
        form = 'primal';
    else
        form = 'dual';
    end
else
    form = param.form;
end

%% Regularization parameter
lambda_x = param.lambda_x;
lambda_y = param.lambda_y;

if ischar(param.lambda_x) && strcmp(param.lambda_x,'auto')
    % Here we use the Ledoit-Wolf method to estimate the regularization
    % parameter analytically.
    % Get samples from each class separately and correct by the class
    % means mu1 and mu2 using bsxfun.
    lambda_x = LedoitWolfEstimate(X, form);
end        
if ischar(param.lambda_y) && strcmp(param.lambda_y,'auto')
    lambda_y = LedoitWolfEstimate(Y, form);
end        

% nLambdaX= numel(param.lambda_x);
% nLambdaY= numel(param.lambda_y);

%% (cross-)covariance matrices of X and Y
Cxx= (X'*X)/N; %cov(X);
Cyy= (Y'*Y)/N; %cov(Y);

% Regularization matrices (scaled identity matrices)
Ip= trace(Cxx)/P * eye(P);
Iq= trace(Cyy)/Q * eye(Q);

% Cross-covariance matrices
Cxy= (X' * Y)/N;
Cyx= Cxy';

%% Regularize covariance
Cxx= (1-lambda_x) * Cxx + lambda_x * Ip;
Cyy= (1-lambda_y) * Cyy + lambda_y * Iq;

%% Cholesky decomposition of Cyy to ensure symmetric EV problem
%% (see Hardoon et al p. 2644)
Ryy= chol(Cyy);
%         iRyy= inv(Ryy);

% We perform a change in coordinates Uy = Ryy * Wy and thus obtain
% a symmetric eigenvalue problem  A*Uy = lambda * Uy
%         A= iRyy' * Cyx * (Cxx_reg \ Cxy) * iRyy;
A= (Ryy' \ Cyx) * (Cxx \ Cxy) / Ryy;

% Enforce symmetry (can be non-symmetric due to numerical problems)
A= 0.5*(A + A');

assert(all(eig(A) > 0), 'A is not positive semidefinite') % debug 

% Ensure symmetric and positive matrix with eig
% https://uk.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
%         A = nearestSPD(A); It may enter in endless loop 
% Using https://github.com/higham/modified-cholesky, which seems to
% work similarly to nearestSPD, but does not enter in endless loop
% [L, D, P, D0] = modchol_ldlt(A); 
% A = P'*L*D*L'*P;      % Modified matrix: symmetric pos def.        

if isempty(param.n) 
    [Uy, D]= eig(A);
else
    [Uy, D]= eigs(A, param.n); % eig returns ALL EVs and it returns them in no particular order
end

% Sort D and Uy in descending order
% if any(any(D<0))
%     disp('N.B.!!! Negative value in matrix D of eigenvalues!!!!  Flipping the sign!!!' );
%     D(D<0)=D(D<0)*-1;
% end
[D,idx] = sort(diag(D),'descend');
D = diag(D);
Uy = Uy(:, idx); 

% Change coordinates back to Wy
Yw = Ryy \ Uy;

%%% Xw's can be obtained from Yws as (Vinod et al paper)

Xw = real((Cxx \ Cxy) * Yw / sqrtm(D));

%% Canonical variates
Xv = X * Xw;
Yv = Y * Yw;

% % If correlation are negative, flip the respective X variables to make them
% % positive
% for ii=1:numel(Rtr{ix,iy})
%     if Rtr{ix,iy}(ii)<0
%         %         warning('Canonical pair #%d yields a negative correlation, flipping the correlation and the respective X components',ii)
%         Rtr{ix,iy}(ii)= -Rtr{ix,iy}(ii);
%         XStr{ix,iy}(:,ii)= -XStr{ix,iy}(:,ii);
%         XWtr{ix,iy}(:,ii)= -XWtr{ix,iy}(:,ii);
%     end
% end

%% add parameters to model struct
model.lambda_x      = lambda_x;
model.lambda_y      = lambda_y;
model.Xw            = Xw;
model.Yw            = Yw;
model.Xv            = Xv;
model.Yv            = Yv;
if param.correlation
    [r,p] = corr(Xv, Yv);
    model.r = diag(r);
    model.p = diag(p);
    assert(all(model.r > 0), 'some correlation values are negative, need to flip covariates?') % debug
end
if param.pattern
    model.Xp            = X' * Xv;
    model.Yp            = Y' * Yv;
end


