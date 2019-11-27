function cf = train_multiclass_lda(param,X,clabel)
% Trains a multi-class linear discriminant analysis (LDA) classifier with 
% shrinkage regularization of the within-class scatter matrix. Multi-class
% LDA can be seen as a prototype classifier: First, the data is mapped
% onto a (C-1)-dimensional discriminative subspace, where C is the number
% of classes. Then, a sample is assigned to the class that has the closest
% class centroid in terms of Euclidean distance. This is equivalent to
% using the Mahalanobis distance metric in the original space.
%
% Note: Use for more than two classes. For two classes, use train_lda.
%
% Usage:
% cf = train_lda_multi(param,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1), 2's (class 2), 3's (class 3) and so on
%
% param          - struct with hyperparameters:
% .reg          - type of regularization
%                 'shrink': shrinkage regularization using (1-lambda)*C +
%                          lambda*nu*I, where nu = trace(C)/P and P =
%                          number of features. nu assures that the trace of
%                          C is equal to the trace of the regularization
%                          term. 
%                 'ridge': ridge-type regularization of C + lambda*I,
%                          where C is the covariance matrix and I is the
%                          identity matrix
%                  (default 'shrink')
% .lambda        - if reg='shrink', the regularization parameter ranges 
%                  from 0 to 1 (where 0=no regularization and 1=maximum
%                  regularization). If 'auto' then the shrinkage 
%                  regularization parameter is calculated automatically 
%                  using the Ledoit-Wolf formula(function cov1para.m). 
%                  If reg='ridge', lambda ranges from 0 (no regularization)
%                  to infinity.
%
%Output:
% cf - struct specifying the classifier with the following fields:
% classifier   - 'multiclass_lda', type of the classifier
% W            - projection matrix with C-1 discriminant directions, where
%                C is the number of classes
%

% (c) Matthias Treder 2018

nclasses = max(clabel);
[nsamples, nfeatures] = size(X);

% Number of samples per class
nc = arrayfun(@(c) sum(clabel == c), 1:nclasses);

%% Calculate sample mean and class centroids
mbar = mean(X);            % sample mean
centroid = zeros(nclasses, nfeatures);       % class means
for c=1:nclasses
    centroid(c,:) = mean(X(clabel==c,:));
end

%% Between-classes scatter for multi-class
Sb = zeros(nfeatures);
for c=1:nclasses
    Sb = Sb + nc(c) * (centroid(c,:)-mbar)'*(centroid(c,:)-mbar);
end

%% Within-class scatter
Sw = zeros(nfeatures);
for c=1:nclasses
    Sw = Sw + (nc(c)-1) * cov(X(clabel==c,:));
end

%% Regularization
lambda = param.lambda;

if strcmp(param.reg,'shrink')
    % SHRINKAGE REGULARIZATION
    if (ischar(lambda)&&strcmp(lambda,'auto'))
        % Here we use the Ledoit-Wolf method to estimate the regularization
        % parameter analytically. 
        % Get samples from each class separately and correct by the class
        % means using bsxfun.
        for c=1:nclasses
            X(clabel==c,:) = bsxfun(@minus,X(clabel==c,:),centroid(c,:));
        end
        lambda= LedoitWolfEstimate(X,'primal');
    end
    % We write the regularized scatter matrix as a convex combination of
    % the empirical scatter Sw and an identity matrix scaled to have
    % the same trace as Sw
    Sw = (1-lambda)* Sw + lambda * eye(nfeatures) * trace(Sw)/nfeatures;

else
    % RIDGE REGULARIZATION
    % The ridge lambda must be provided directly as a number
    Sw = Sw + lambda * eye(nfeatures);
end

%% Solve generalized eigenvalue problem to obtain discriminative subspace
[W,D] = eig(Sb, Sw, 'vector');
[~, so] = sort(D,'descend');
W = W(:,so(1:min(nclasses, nfeatures+1)-1));

% Columns of W need to be scaled correctly such that it turns the 
% covariance matrix (ie Sw/(N-1) ) into identity
W  = W * diag(1./sqrt(diag(W'*Sw*W)/(nsamples-1)));

%% Set up classifier struct
cf= struct('classifier','multiclass_lda','W',W,'lambda',lambda,'nclasses',nclasses);

% Map the class centroids onto the discriminative subspace for later
% prototype classification
cf.centroid = centroid * cf.W;

