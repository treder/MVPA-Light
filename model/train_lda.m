function [cf,Sw,lambda,mu1,mu2] = train_lda(param,X,clabel)
% Trains a linear discriminant analysis with regularization of the 
% covariance matrix.
%
% Usage:
% cf = train_lda(param,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% param          - struct with hyperparameters:
% .reg          - type of regularization
%                 'shrink': shrinkage regularization using (1-lambda)*Sw +
%                          lambda*nu*I, where Sw is the scatter matrix,
%                          I is the identity matrix, nu = trace(Sw)/P and 
%                          P = number of features. nu assures that the trace of
%                          Sw is equal to the trace of the regularization
%                          term. 
%                 'ridge': ridge-type regularization using Sw + lambda*I
%                  (default 'shrink')
% .lambda        - if reg='shrink', the regularization parameter ranges 
%                  from 0 to 1 (where 0=no regularization and 1=maximum
%                  regularization). If 'auto' then the shrinkage 
%                  regularization parameter is calculated automatically 
%                  using the Ledoit-Wolf formula(function cov1para.m). 
%                  If reg='ridge', lambda ranges from 0 (no regularization)
%                  to infinity.
%                  If multiple values are given, a grid search is performed
%                  for the best lambda (only for reg='shrink')
% .prob          - if 1, decision values are returned as probabilities. If
%                  0, the decision values are simply the distance to the
%                  hyperplane. Calculating probabilities takes more time
%                  and memory so don't use this unless needed. Probabilities 
%                  can be unreliable in high dimensions (default 0). Note
%                  that for probabilities the covariance matrix needs to be
%                  calculated, so it can only be used in primal form (hence
%                  set hyperparameter.form = 'primal')
% .scale         - if 1, the projection vector w is scaled such that the
%                  mean of class 1 (on the training data) projects onto +1
%                  and the mean of class 2 (on the training data) projects
%                  onto -1
% .form          - uses the 'primal' or 'dual' form of the solution to
%                  determine w. If 'auto', auomatically chooses the most 
%                  efficient form depending on whether #samples > #features
%                  (default 'auto'). 'primal' corresponds to the standard
%                  LDA approach using the features x features covariance
%                  matrix. 'dual' uses the samples x samples Gram matrix
%                  instead. In both cases, w is calculated.
%
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold
%
% The following output arguments can be returned optionally:
% Sw           - covariance matrix (possibly regularized)
% mu1,mu2      - class means
% n            - total number of samples

% (c) Matthias Treder

[n, p] = size(X);
cf= struct();

ix1= (clabel==1);  % logical indices for samples in class 1
ix2= (clabel==2);  % logical indices for samples in class 2

n1 = sum(ix1);
n2 = sum(ix2);

% Get class means
mu1= mean(X(ix1,:))';
mu2= mean(X(ix2,:))';

%% Choose between primal and dual form
if strcmp(param.form, 'auto')
    if n >= p
        form = 'primal';
    else
        form = 'dual';
    end
else
    form = param.form;
end

%% Regularization parameter
lambda = param.lambda;

if strcmp(param.reg,'shrink')
    % SHRINKAGE REGULARIZATION
    if ischar(lambda) && strcmp(lambda,'auto')
        % Here we use the Ledoit-Wolf method to estimate the regularization
        % parameter analytically.
        % Get samples from each class separately and correct by the class
        % means mu1 and mu2 using bsxfun.
        lambda = LedoitWolfEstimate([bsxfun(@minus,X(ix1,:),mu1');bsxfun(@minus,X(ix2,:),mu2')], form);
        
    elseif numel(lambda)==1
        % Shrinkage parameter is given directly as a number. Don't need to
        % do anything here
        
    elseif ~ischar(lambda) && numel(lambda)>1
        if strcmp(form,'dual'), error('hyperparameter grid search for LDA is currently only implemented for the primal form'), end
        tune_hyperparameter_lda;
    end
end

%% PRIMAL FORM
if strcmp(form, 'primal')

    % Calculate common covariance matrix (actually its unscaled version aka
    % within-class scatter matrix).
    % It should be weighted by the relative class proportions
    Sw= n1 * cov(X(ix1,:),1) + n2 * cov(X(ix2,:),1);
    
    % Regularization
    if strcmp(param.reg,'shrink')
        Sw = (1-lambda)* Sw + lambda * eye(p) * trace(Sw)/p;
    else
        % RIDGE REGULARIZATION
        % The ridge lambda must be provided directly as a number
        Sw = Sw + lambda * eye(p);
    end
    
    % Classifier weight vector (= normal to the separating hyperplane)
    cf.w = Sw\(mu1-mu2);
    
%% DUAL FORM
elseif strcmp(form, 'dual')
    % the dual formulation is essentially the same as 
    % in kernel FDA
    
    % remove grand mean (this is done after calculating class means since it's mainly
    % needed for computing the covariance/Gram matrices)
    X = X - repmat(mean(X,1), n, 1);
    
    % Gram matrix
    K = X*X';
    
    %% N: "Dual" of within-class scatter matrix
    N = K(:,ix1) * (eye(n1) - 1/n1) * K(ix1,:) + K(:,ix2) * (eye(n2) - 1/n2) * K(ix2,:);

    % Regularization - this assures equivalence to the primal approach
    % Note that for the primal: w' (Sw + lambda I) w
    % we get the dual: alpha' (N + lambda K) alpha
    % for ridge regularizarion. An analogous result is obtained for shrinkage. 
    % This means that the regularization target is K (not I, as in the primal case).
    if strcmp(param.reg,'shrink')
        N = (1-lambda)* N + lambda * K * trace(K)/p;
    else
        % ridge regularization
        N = N + lambda * K;
    end

    % Since the regularization target above is K, N is generally still
    % ill-conditioned. Here we fix this here by adding a little bit of an
    % identity matrix
    N = N + param.lambda_n * trace(N) * eye(n)/n;

    %% M: "Dual" of class means
    Mu1 = mean( K(:, ix1), 2);
    Mu2 = mean( K(:, ix2), 2);

    %% get dual weights
    cf.alpha = N\(Mu1 - Mu2);
    
    %% translate into weight vector
    cf.w = X' * cf.alpha;

end

%% Scale w such that the class means are projected onto +1 and -1
if param.scale
    cf.w = cf.w / ((mu1-mu2)'*cf.w) * 2;
end

%% Bias term 
% Determines the classification threshold
cf.b= -cf.w'*(mu1+mu2)/2;

%% Set up classifier struct
cf.prob     = param.prob;
cf.lambda   = lambda;

if param.prob == 1
    % If probabilities are to be returned as decision values, we need to
    % determine the priors and also save the covariance matrix and the class
    % means. This consumes extra time and memory so keep prob = 0 unless 
    % you need it.
    % The goal is to calculate posterior probabilities (probability for a 
    % sample to be class 1).
    
    % The prior probabilities are calculated from the training
    % data using the proportion of samples in each class
    cf.prior1 = n1/n;
    cf.prior2 = n2/n;

    cf.Sw   = Sw;
    cf.mu1  = mu1;
    cf.mu2  = mu2;
    cf.n    = n;
    
end
