function [cf,C,lambda,mu1,mu2] = train_lda(cfg,X,clabel)
% Trains a linear discriminant analysis with (optional) shrinkage
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
% .lambda        - regularisation parameter between 0 and 1 (where 0 means
%                   no regularisation and 1 means full max regularisation).
%                   If 'auto' then the regularisation parameter is
%                   calculated automatically using the Ledoit-Wolf formula(
%                   function cov1para.m)
% .prob          - if 1, probabilities are returned as decision values. If
%                  0, the decision values are simply the distance to the
%                  hyperplane. Calculating probabilities takes more time
%                  and memory so don't use this unless needed (default 0)
% .scale         - if 1, the projection vector w is scaled such that the
%                  mean of class 1 (on the training data) projects onto +1
%                  and the mean of class 2 (on the training data) projects
%                  onto -1
%
%Output:
% cf - struct specifying the classifier with the following fields:
% classifier   - 'lda', type of the classifier
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold
%
% The following fields can be returned optionally:
% C            - covariance matrix (possibly regularised)
% mu1,mu2      - class means
%

% (c) Matthias Treder 2017

idx1= (clabel==1);  % logical indices for samples in class 1
idx2= (clabel==2);  % logical indices for samples in class 2

N1 = sum(idx1);
N2 = sum(idx2);
N= N1 + N2;

% Calculate common covariance matrix
% Should be weighted by the relative class proportions
C= N1/N * cov(X(idx1,:)) + N2/N * cov(X(idx2,:));

% Get class means
mu1= mean(X(idx1,:))';
mu2= mean(X(idx2,:))';

lambda = cfg.lambda;
    
% Regularise covariance matrix using shrinkage
if (ischar(lambda)&&strcmp(lambda,'auto'))
    % Here we use the Ledoit-Wolf method to estimate the regularisation
    % parameter analytically.
    % Get samples from each class separately and correct by the class
    % means mu1 and mu2 using bsxfun.
    [C, lambda]= cov1para([bsxfun(@minus,X(idx1,:),mu1');bsxfun(@minus,X(idx2,:),mu2')]);
    
elseif numel(lambda)==1
    % Shrinkage parameter is given directly as a number.
    % We write the regularised covariance matrix as a convex combination of
    % the empirical covariance C and an identity matrix scaled to have
    % the same trace as C
    C = (1-lambda)* C + lambda * eye(size(C,1)) * trace(C)/size(X,2);
    
elseif ~ischar(lambda) && numel(lambda)>1
    % Multipe lambdas given: perform tuning using a grid search
    tune_hyperparameter_lda;
end

% Classifier weight vector (= normal to the separating hyperplane)
w = C\(mu1-mu2);

% Scale w such that the class means are projected onto +1 and -1
if cfg.scale
    w = w / ((mu1-mu2)'*w) * 2;
end

% Bias term determining the classification threshold
b= w'*(mu1+mu2)/2;

%% Prepare output
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

    cf.C = C;
    cf.mu1 = mu1;
    cf.mu2 = mu2;
    
    % Projected standard deviation
%     cf.sigma = sqrt(w' * C * w);
%
%     % Projected class means
%     cf.m1 = w' * mu1;
%     cf.m2 = w' * mu2;
end
