function [clabel, dval, prob] = test_naive_bayes2(cf,X)
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


dval = arrayfun( @(c) -( bsxfun(@minus, X, cf.class_means(c,:,:)) .^2) ./ cf.var(c*ones(size(X,1),1),:,:), 1:cf.nclasses, 'Un',0);
 
if isempty(cf.neighbours)
  dval = cell2mat( cellfun(@(d) sum(d,2)/2, dval, 'Un',0));
else
  N1 = sparse(double(cf.neighbours{1}));
  N2 = sparse(double(cf.neighbours{2}));
  
  dval = cat(1, dval{:});
  dvalnew = zeros(size(dval,1), size(N1,1)*size(N2,1));
  
  for k = 1:size(dval,1)
    dvalnew(k,:) = reshape(N1*reshape(dval(k,:),[size(N1,2) size(N2,2)])*N2',1,[]);
  end
  clear dval;
  dval = reshape(dvalnew, size(X,1), cf.nclasses, size(dvalnew,2));
  clear dvalnew;
end

% add prior
for k = 1:cf.nclasses
  dval(:,k,:) = dval(:,k,:) + cf.prior(k);
end

% For each sample, find the closest centroid and assign it to the
% respective class
[~, clabel] = max(dval, [], 2); % this avoids an expensive for-loop

if nargout>2
    % apply softmax to obtain the posterior probabilities
    prob = exp(dval);
    Px = sum(prob,2);
    for cc=1:cf.nclasses
        prob(:,cc,:) = prob(:,cc,:) ./ Px;
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
