function [cf,Sw,lambda,mu1,mu2] = train_lda(cfg,X,clabel)
% Trains a linear discriminant analysis with (optional) 
% regularisation of the covariance matrix.
%
% Usage:
% cf = train_lda(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with hyperparameters:
% .reg          - type of regularisation
%                 'shrink': shrinkage regularisation using (1-lambda)*C +
%                          lambda*nu*I, where nu = trace(C)/P and P =
%                          number of features. nu assures that the trace of
%                          C is equal to the trace of the regularisation
%                          term. 
%                 'ridge': ridge-type regularisation of C + lambda*I,
%                          where C is the covariance matrix and I is the
%                          identity matrix
%                  (default 'shrink')
% .lambda        - if reg='shrink', the regularisation parameter ranges 
%                  from 0 to 1 (where 0=no regularisation and 1=maximum
%                  regularisation). If 'auto' then the shrinkage 
%                  regularisation parameter is calculated automatically 
%                  using the Ledoit-Wolf formula(function cov1para.m). 
%                  If reg='ridge', lambda ranges from 0 (no regularisation)
%                  to infinity.
%                  If multiple values are given, a grid search is performed
%                  for the best lambda (only for reg='shrink')
% .prob          - if 1, decision values are returned as probabilities. If
%                  0, the decision values are simply the distance to the
%                  hyperplane. Calculating probabilities takes more time
%                  and memory so don't use this unless needed. Probabilities 
%                  can be unreliable in high dimensions (default 0)
% .scale         - if 1, the projection vector w is scaled such that the
%                  mean of class 1 (on the training data) projects onto +1
%                  and the mean of class 2 (on the training data) projects
%                  onto -1
%
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold
%
% The following fields can be returned optionally:
% Sw           - covariance matrix (possibly regularised)
% mu1,mu2      - class means
% N            - total number of samples

% (c) Matthias Treder 2017

idx1= (clabel==1);  % logical indices for samples in class 1
idx2= (clabel==2);  % logical indices for samples in class 2

N1 = sum(idx1);
N2 = sum(idx2);
N= N1 + N2;

% Calculate common covariance matrix (actually its unscaled version aka
% within-class scatter matrix).
% It should be weighted by the relative class proportions
Sw= N1 * cov(X(idx1,:),1) + N2 * cov(X(idx2,:),1);

% Get class means
mu1= mean(X(idx1,:))';
mu2= mean(X(idx2,:))';

lambda = cfg.lambda;
    
%% Regularisation
if strcmp(cfg.reg,'shrink')
    % SHRINKAGE REGULARISATION
    if ischar(lambda) && strcmp(lambda,'auto')
        % Here we use the Ledoit-Wolf method to estimate the regularisation
        % parameter analytically.
        % Get samples from each class separately and correct by the class
        % means mu1 and mu2 using bsxfun.
        [~, lambda]= cov1para([bsxfun(@minus,X(idx1,:),mu1');bsxfun(@minus,X(idx2,:),mu2')]);
        
    elseif numel(lambda)==1
        % Shrinkage parameter is given directly as a number. Don't need to
        % do anything here
        
    elseif ~ischar(lambda) && numel(lambda)>1
        tune_hyperparameter_lda;
    end
    
    % We write the regularised scatter matrix as a convex combination of
    % the empirical scatter Sw and an identity matrix scaled to have
    % the same trace as Sw
    Sw = (1-lambda)* Sw + lambda * eye(size(Sw,1)) * trace(Sw)/size(X,2);
else
    % RIDGE REGULARISATION
    % The ridge lambda must be provided directly as a number
    Sw = Sw + lambda * eye(size(Sw,1));
end

% Classifier weight vector (= normal to the separating hyperplane)
w = Sw\(mu1-mu2);

% Scale w such that the class means are projected onto +1 and -1
if cfg.scale
    w = w / ((mu1-mu2)'*w) * 2;
end

% Bias term determining the classification threshold
b= -w'*(mu1+mu2)/2;

%% Set up classifier struct
cf= struct('w',w,'b',b,'prob',cfg.prob,'lambda',lambda);

if cfg.prob == 1
    % If probabilities are to be returned as decision values, we need to
    % determine the priors and also save the covariance matrix and the cleass
    % means. This consumes extra time and memory so keep prob = 0 unless 
    % you need it.
    % The goal is to calculate posterior probabilities (probability for a 
    % sample to be class 1).
    
    % The prior probabilities are calculated from the training
    % data using the proportion of samples in each class
    cf.prior1 = N1/N;
    cf.prior2 = N2/N;

    cf.Sw   = Sw;
    cf.mu1  = mu1;
    cf.mu2  = mu2;
    cf.N    = N;
    
end
