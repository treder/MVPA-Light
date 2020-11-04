function [pparam, X, clabel] = mv_preprocess_pca(pparam, X, clabel)
% Performs Principal Component Analysis (PCA) by projecting the data onto
% the n leading eigenvectors of the covariance matrix.
%
%Usage:
% [pparam, X, clabel] = mv_preprocess_pca(pparam, X, clabel)
%
%Parameters:
% X              - [... x ... x ...] data matrix
% clabel         - [samples x 1] vector of class labels
%
% pparam         - [struct] with preprocessing parameters
% .n                - number of principal components to keep (default 20)
% .target_dimension - dimension along which the covariance matrix is
%                     calculated (eg time or sample dimension) (default 3)
% .feature_dimension - which dimension codes the features (eg the channels)
%                     (default 2)
% .normalize        - if 1, all PCs are scaled to have variance 1 (default
%                     1). This only works if X has a rank of at least n. 
%
% Note: features x features covariance matrices are calculated across the
% target dimension. A covariance matrix is calculated for every element of
% any other dimension (if any) and an average covariance matrix is
% calculated. For instance, if the data is [trials x channels x times] (and
% target_dimension=3, feature_dimension=2) then single-trial covariance
% matrices are calculated across time and then averaged across trials. 
% If the data is just [samples x features], target_dimension should be set
% to 1.
%
% Nested preprocessing: Eigenvectors of the covariance matrix are
% calculated on the train data. Both train and test data are projected on
% the leading n eigenvectors. 

f = pparam.feature_dimension;
nd = ndims(X);

if pparam.is_train_set
    
    t = pparam.target_dimension;

    % all other dimensions serve as 'search' dimensions: covariance is
    % accumulated over these dimensions
    search_dim = setdiff(1:nd, [t, f]);
    
    % Create all combinations of elements in the search dimensions for looping
    if isempty(search_dim)
        % no search dimensions, we just perform cross-validation once
        dim_loop = {':'};
    else
        sz_search = size(X);
        sz_search = sz_search(search_dim);
        len_loop = prod(sz_search);
        dim_loop = zeros(nd, len_loop);
        for rr = 1:numel(sz_search)  % row
            seq = mv_repelem(1:sz_search(rr), prod(sz_search(1:rr-1)));
            dim_loop(search_dim(rr), :) = repmat(seq, [1, len_loop/numel(seq)]);
        end
        % to use dim_loop for indexing, we need to convert it to a cell array
        dim_loop = num2cell(dim_loop);
        % we only need to replace the feature and target rows with {:}
        % operators 
        dim_loop(f,:) = {':'};
        dim_loop(t,:) = {':'};
    end
    
    % Start with an empty covariance matrix
    C = zeros(size(X, f));
    
    % if the feature dimension comes before the target dimension, we have
    % to flip the matrix for the covariance calculation
    if f < t
        for ix = dim_loop
            C = C + cov(squeeze(X(ix{:},:))');
        end
    else
        for ix = dim_loop
            C = C + cov(squeeze(X(ix{:},:)) );
        end
    end
    
    C = C / size(dim_loop,2);
    
    % Eigenvector decomposition of the pooled covariance matrix
    [V, D] = eig(C);
    D = diag(D);
    
    % Sort according to descending eigenvalues
    [D,ind] = sort(D, 'descend');
    V = V(:,ind);
    
    % Select n leading components
    D = D(1:pparam.n);
    V = V(:, 1:pparam.n);

    % Scale V such that components have unit variance 
    if pparam.normalize
        V = V * diag(1./sqrt(D));
    end
    
    pparam.V = V;
end

% reshape data matrix such that it is [... x features] because then we can
% project it onto the principal components with a single matrix
% multiplication. To this end, first permute the features dimension to be
% the last dimension, then reshape
pos = 1:nd;
pos(f) = nd;
pos(nd) = f;
X = permute(X, pos);
sz = size(X); % remember size
X = reshape(X, [], size(X, nd));

% Project onto eigenvectors
X = X * pparam.V;

% Undo reshape
sz(end) = pparam.n;
X = reshape(X, sz);

% permute feature dimension back to its original position
X = permute(X, pos);
