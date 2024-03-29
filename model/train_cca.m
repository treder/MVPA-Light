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
% .n          - number of components
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
%
%Output:
% model - struct specifying the regression model with the following fields:
% wx            - matrix of weight vectors for X 
% wy            - matrix of weight vectors for Y
%
%Reference:
% HD Vinod. "Canonical ridge and econometrics of joint production".
% Journal of Econometrics, Volume 4, Issue 2, May 1976, Pages 147-166
%

N = size(X,1);
P = size(X,2);
Q = size(Y,2);

assert(size(X,1)==size(Y,1), 'X and Y must have the same number of samples')
assert(param.n <= min(P,Q), 'n must be smaller or equal to the number of features in X or Y')


% csa_setDefault(param,'svd', 0);
% csa_setDefault(param,'nSvd', max([N,P,Q]));

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
lambda = param.lambda;

if ischar(param.lambda_x) && strcmp(param.lambda_x,'auto')
    % Here we use the Ledoit-Wolf method to estimate the regularization
    % parameter analytically.
    % Get samples from each class separately and correct by the class
    % means mu1 and mu2 using bsxfun.
    lambda = LedoitWolfEstimate(X, form);
end        

%%


if ischar(param.lambda_x) && strcmp(param.lambda_x,'auto')
    [~, param.lambda_x] = cov1para(X);
%     [~, opt.lambda_x] = stats_shrinkage_cov(X,'rblw');
end

if ischar(param.lambda_y) && strcmp(param.lambda_y,'auto')
    [~, param.lambda_y] = cov1para(Y);
%     [~, opt.lambda_y] = stats_shrinkage_cov(Y,'rblw');
end

nLambdaX= numel(param.lambda_x);
nLambdaY= numel(param.lambda_y);

%% Get (cross-)covariance matrices of X and Y
Cxx= (X'*X)/N; %cov(X);
Cyy= (Y'*Y)/N; %cov(Y);

% Regularization matrices (scaled identity matrices)
Ip= trace(Cxx)/P * eye(P);
Iq= trace(Cyy)/Q * eye(Q);

% Cross-covariance matrices
Cxy= (X' * Y)/N;
Cyx= Cxy';

more=struct();
XWtr= cell(nLambdaX,nLambdaY);
YWtr= cell(nLambdaX,nLambdaY);
XStr= cell(nLambdaX,nLambdaY);
YStr= cell(nLambdaX,nLambdaY);
more.XL= cell(nLambdaX,nLambdaY);
more.YL= cell(nLambdaX,nLambdaY);
Rtr= cell(nLambdaX,nLambdaY);

%% Run CCA 
% warning('Start looping over lambdas\n');

for ix=1:nLambdaX
    for iy=1:nLambdaY
%         warning('Iteration x=%d, y=%d [dim(X)=%d, dim(Y)=%d, lambda_x=%0.6f, lambda_y=%0.6f]\n',ix,iy,P,Q,opt.lambda_x(ix),opt.lambda_y(iy));
        Cxx_reg= (1-param.lambda_x(ix)) * Cxx + param.lambda_x(ix) * Ip;
        Cyy_reg= (1-param.lambda_y(iy)) * Cyy + param.lambda_y(iy) * Iq;
        
        %% Cholesky decomposition of Cyy to ensure symmetric EV problem
        %% (see Hardoon et al p. 2644)
        Ryy= chol(Cyy_reg);
%         iRyy= inv(Ryy);
        
        % We perform a change in coordinates Uy = Ryy * Wy and thus obtain
        % a symmetric eigenvalue problem  A*Uy = lambda * Uy
%         A= iRyy' * Cyx * (Cxx_reg \ Cxy) * iRyy;
        A= (Ryy' \ Cyx) * (Cxx_reg \ Cxy) / Ryy;

        
        % Enforce symmetry (can be non-symmetric due to numerical problems)
        A= 0.5*(A + A');

        
        % Ensure symmetric and positive matrix with eig
        % https://uk.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        %         A = nearestSPD(A); It may enter in endless loop 
        % Using https://github.com/higham/modified-cholesky, which seems to
        % work similarly to nearestSPD, but does not enter in endless loop
        [L, D, P, D0] = modchol_ldlt(A); 
        A = P'*L*D*L'*P;      % Modified matrix: symmetric pos def.        
        
        if isempty(param.nComp) 
%             [Uy, D]= eig(double(A));
            [Uy, D]= eig(A);
        else
%             [Uy, D]= eigs(double(A), opt.nComp); % eig returns ALL EVs and it returns them in no particular order
            [Uy, D]= eigs(A, param.nComp); % eig returns ALL EVs and it returns them in no particular order
        end
        % Sort D and Uy in descending order
        if any(any(D<0))
            disp('N.B.!!! Negative value in matrix D of eigenvalues!!!!  Flipping the sign!!!' );
            D(D<0)=D(D<0)*-1;
        end
        [D,idx] = sort(diag(D),'descend');
        D = diag(D);
        Uy = Uy(:, idx); 
        
        % Change coordinates back to Wy
        YWtr{ix,iy} = Ryy \ Uy;
        
        %% Get CCA coefficients (following formula from the Vinod paper)
        %%% Get Yw's as eigenvectors
%         MAT=  (Cyy_reg \ Cyx) * (Cxx_reg \ Cxy);
%         MAT= 0.5 * ( MAT + MAT');  % ensure that it's symmetric (can be slightly asymemtric due to neumerical issues I think)
%         [Uy,D]= eigs(MAT,opt.nComp);
%         
%         % Sort D and Uy in descending order
%         [D,idx] = sort(diag(D),'descend');
%         D = diag(D);
%         Uy = Uy(:, idx); 
%         Yw_eigs{ix,iy} =Uy;
        
        %%% Xw's can be obtained from Yws as (Vinod et al paper)
        
        XWtr{ix,iy}=  real((Cxx_reg \ Cxy) * YWtr{ix,iy} / sqrtm(D));
        
        %% Canonical variates
        XStr{ix,iy}= X*XWtr{ix,iy};
        YStr{ix,iy}= Y*YWtr{ix,iy};
%         Ys_eigs{ix,iy}= Y*Yw_eigs{ix,iy};
        %% Canonical correlations
        [rval,pval] = corr(XStr{ix,iy},YStr{ix,iy});
        Rtr{ix,iy}  = diag(rval);
        Ptr{ix,iy}  = diag(pval);
        
        % If correlation are negative, flip the respective X variables to make them
        % positive
        for ii=1:numel(Rtr{ix,iy})
            if Rtr{ix,iy}(ii)<0
                %         warning('Canonical pair #%d yields a negative correlation, flipping the correlation and the respective X components',ii)
                Rtr{ix,iy}(ii)= -Rtr{ix,iy}(ii);
                XStr{ix,iy}(:,ii)= -XStr{ix,iy}(:,ii);
                XWtr{ix,iy}(:,ii)= -XWtr{ix,iy}(:,ii);
            end
        end
        
        %% Canonical loadings
        more.XLtr{ix,iy} = X' * XStr{ix,iy}; %Xorig' * Xs{ix,iy};
        more.YLtr{ix,iy} = Y' * YStr{ix,iy}; %Yorig' * Ys{ix,iy};
        more.XL{ix,iy}   = more.XLtr{ix,iy};
        more.YL{ix,iy}   = more.YLtr{ix,iy};
    end
    
end

