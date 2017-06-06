function cfy = train_logreg(X,labels,param)
% Trains a logistic regression classifier. Uses PCA for dimension reduction
% and regularisation.
%
% Usage:
% cf = train_logreg(X,labels,param)
% 
%Parameters:
% X              - [number of samples x number of features] matrix of
%                  training samples
% labels         - [number of samples] vector of class labels containing 
%                  1's (class 1) and -1's (class 2)
% param          - struct with hyperparameters:
% .nPC           - number of principal components to retain
%                                OR
% .fracVar       - the number of PCs is selected such that this fraction 
%                  of variance is retained. Either nPC or fracVar should be
%                  set, not both.
%
%
% Note that lambda can also be directly specified by setting params to the
% lambda value.
%
%Output:
% cf - struct specifying the classifier with the following fields:
% classifier   - 'lda', type of the classifier
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold 
% C            - covariance matrix (possibly regularised)
% mu1,mu2      - class means
% N1,N2        - number of samples in classes 1 and 2
%

if ~exist('cfg','var') || isempty(param)
    param.lambda = 0;
elseif ~isstruct(param)
    % lambda was provided directly
    param.lambda= param;
end

%% Train logistic regression using GLMFIT
beta= glmfit(X, labels(:)>0 ,'binomial','link','logit');

% beta contains the intercept (beta(1)) and the projection vector w in beta(2:end)

% TODO: correcting bias for unbalanced classes

%% Prepare output
cfy= struct();
cfy.classifier= 'Logistic Regression';
cfy.w= beta(2:end);
cfy.b= beta(1);
