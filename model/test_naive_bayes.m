function [clabel, dval, prob] = test_naive_bayes(cf,X)
% Applies a Naive Bayes classifier to test data.
%
% Usage:
% clabel = test_naive_bayes(cf, X)
% 
%Parameters:
% cf             - classifier. See train_naive_bayes
% X              - [samples x features] matrix of test data or 
%                  [samples x ... x ... x features] matrix with additional
%                   dimensions (useful in searchlight analysis)
%                  additional dimensions
%
%Output:
% clabel     - predicted class labels
% dval       - decision values for the winning class (they are not
%              distances to the hyperplane)
% prob       - posterior probabilities for the winning class (given as normalized dvals)
%
% Note: For more than 2 classes,
% dval and prob comes as matrices and hence should not be used
% within the high-level functions. They can be used when test_naive_bayes
% is called by hand.
dval = arrayfun( @(c) -( bsxfun(@minus, X, cf.class_means(c,:,:,:,:,:,:)) .^2) ./ repmat(cf.var(c,:,:,:,:,:,:), [size(X,1) ,1]), 1:cf.nclasses, 'Un',0);


% We need some caution in dealing with data with extra dimensions
% (typically used in searchlight):
% From the data size alone it is unclear whether a dimension is a
% searchlight dimension or whether it represents features. For instance,
% a 10 x 20 x 30 matrix might have 10 samples, a searchlight dimension of
% size 20, and 30 features, but it might also have 10 samples, and two
% searchlight dimensions of size 20 and 30.
% This is relevant because we have to sum across features (the resultant
% predictions have dimensions 10 x 20) but not across searchlight dimensions 
% (the resultant prediction have dimensions 10 x 20 x 30).
% The parameter is_multivariate denotes whether the last dimension should be
% considered as features and summed over (is_multivariate = 1) or whether it
% not (is_multivariate = 0). Note that the high-level function mv_classify 
% sets is_multivariate automatically, the user only needs to set it when
% using the train/test functions manually and using extra dimensions for
% searchlight.
if cf.is_multivariate
    % last dimension represents features: sum dvals across all features
    dval = cellfun(@(d) sum(d, ndims(d))/2, dval, 'Un',0);
end

% the 2nd dimension of dval will be the number of classes
if isempty(cf.neighbours)
    if ~isvector(dval{1})
        % insert singleton dimension as 2nd dimension
        dval = cellfun(@(d) reshape(d, [size(d,1) 1 size(d,2) size(d,3) size(d,4) size(d,5)  size(d,6)]), dval, 'Un',0);
    end
    % concatenate along 2nd dimension
    dval = cat(2, dval{:});
else
    N = cell(1,numel(cf.neighbours));
    for k = 1:numel(cf.neighbours)
        N{k} = sparse(double(cf.neighbours{k}));
    end
  
    dim_rows = cellfun('size', N, 1); % dim_rows and dim_cols could be different
    dim_cols = cellfun('size', N, 2);
    assert(isequal(prod(dim_cols), prod(size(shiftdim(dval{1}(1,:,:,:,:))))), 'neighbours specification does not match the dimension of the features');
  
    dval    = cat(1, dval{:});
    dvalnew = zeros([size(dval,1), dim_rows]);
  
    % do the 'smoothing' as matrix multiplication with neighbourhood
    % matrices, which avoids an expensive for-loop in the caller function
    if numel(dim_cols)==1
        dim_cols_shape = [dim_cols 1];
    else
        dim_cols_shape = dim_cols;
    end
    for k = 1:size(dval,1)
        dval_ = reshape(dval(k,:), dim_cols_shape);
        switch numel(dim_cols)
            case 1
                dvalnew(k,:,:,:) = N{1} * dval_;
            case 2
                dvalnew(k,:,:,:) = N{1} * dval_ * N{2}';
            case 3
                for m = 1:dim_rows(1)
                    dvalnew(k,m,:,:) = N{1}(m,:) * (N{2}*shiftdim(dval_(m,:,:,:),1) * N{3}');
                end
            otherwise
                error('neighbours on more than 3 dimensions is currently not supported')
        end
    end
    clear dval;
    dval = reshape(dvalnew, [size(X,1), cf.nclasses, dim_rows]);
    clear dvalnew;
end

% add prior
dval = bsxfun(@plus, dval, cf.prior);

% For each sample, find the closest centroid and assign it to the
% respective class
[~, clabel] = max(dval, [], 2);
clabel = squeeze(clabel);

if nargout>2
    % apply softmax to obtain the posterior probabilities
    prob = exp(dval);
    Px = sum(prob, 2);
    prob = bsxfun(@rdivide, prob, Px);
end

if cf.nclasses == 2
    % for the special case of 2 classes we can rewrite the dvals and probs
    % into single vectors dval1 - dval2 and prob(class=1)
    dval = dval(:,1) - dval(:,2);
    if nargout>2
        prob = prob(:,1);
    end
end
