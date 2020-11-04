function [clabel, dval, prob] = test_naive_bayes(cf,X)
% Applies a Naive Bayes classifier to test data.
%
% Usage:
% clabel = test_naive_bayes(cf, X)
% 
%Parameters:
% cf             - classifier. See train_naive_bayes
% X              - [samples x features] matrix of test data
%
%Output:
% clabel     - predicted class labels
% dval       - decision values for the winning class (they are not
% distances to the hyperplane)
% prob       - posterior probabilities for the winning class (given as normalized dvals)
%
% Note: For more than 2 classes,
% dval and prob comes as matrices and hence should not be used
% within the high-level functions. They can be used when test_naive_bayes
% is called by hand.
dval = arrayfun( @(c) -( bsxfun(@minus, X, cf.class_means(c,:,:,:)) .^2) ./ repmat(cf.var(c,:,:,:), [size(X,1) ,1]), 1:cf.nclasses, 'Un',0);

if ~isfield(cf, 'neighbours') || isempty(cf.neighbours)
    dval = cell2mat( cellfun(@(d) sum(d,2)/2, dval, 'Un',0));
else
    n = numel(cf.neighbours);
    N = cell(1,n);
    for k = 1:n
        N{k} = sparse(double(cf.neighbours{k}));
    end
  
    dim1 = cellfun('size', N, 1); % dim1 and dim2 could be different
    dim2 = cellfun('size', N, 2);
    assert(isequal(dim2, size(shiftdim(X(1,:,:,:,:)))), 'neighbours specification does not match the dimension of the features');
  
    dval    = cat(1, dval{:});
    dvalnew = zeros([size(dval,1), dim1]);
  
    % do the 'smoothing' as matrix multiplication with neighbourhood
    % matrices, which avoids an expensive for-loop in the caller function
    for k = 1:size(dval,1)
        dval_ = reshape(dval(k,:), dim2);
        switch numel(dim2)
            case 1
                dvalnew(k,:,:,:) = N{1}*dval_;
            case 2
                dvalnew(k,:,:,:) = N{1}*dval_*N{2}';
            case 3
                for m = 1:dim1(1)
                    dvalnew(k,m,:,:) = N{1}(m,:)*(N{2}*shiftdim(dval_(m,:,:,:),1)*N{3}');
                end
        otherwise
            error('neighbours on more than 3 dimensions is currently not supportedd')
        end
    end
    clear dval;
    dval = reshape(dvalnew, [size(X,1), cf.nclasses, dim1]);
    clear dvalnew;
end

% add prior
dval = bsxfun(@plus, dval, cf.prior);

% For each sample, find the closest centroid and assign it to the
% respective class
%clabel = zeros(size(X,1),1);
%for ii=1:size(X,1)
%    [~, clabel(ii)] = max(dval(ii,:));
%end
[~, clabel] = max(dval, [], 2);

if nargout>2
    % apply softmax to obtain the posterior probabilities
    prob = exp(dval);
    Px = sum(prob,2);
    for cc=1:cf.nclasses
        prob(:,cc) = prob(:,cc) ./ Px;
    end
end

if cf.nclasses == 2
    % for the special case of 2 classes we can rewrite the dvals and probs
    % into single vectors dval1 - dval2 and prob(class=1)
    dval = dval(:,1) - dval(:,2);
    if nargout>2
        prob = prob(:,1);
    end
end
