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
% Note: dval and prob comes as matrices and hence should not be used
% withint the high-level functions. They can be used when test_naive_bayes
% is called by hand.
dval = arrayfun( @(c) exp(-(bsxfun(@minus, X, cf.class_means(c,:)).^2)./cf.var(c,:)/2), 1:cf.nclasses, 'Un',0);
dval = cell2mat( cellfun(@(d) sum(d,2), dval, 'Un',0));

% add prior
dval = bsxfun(@plus, dval, cf.prior);

% For each sample, find the closest centroid and assign it to the
% respective class
clabel = zeros(size(X,1),1);
for ii=1:size(X,1)
    [~, clabel(ii)] = max(dval(ii,:));
end

if nargout>2
    % apply softmax to obtain the posterior probabilities
    prob = exp(dval);
    Px = sum(prob,2);
    for cc=1:cf.nclasses
        prob(:,cc) = prob(:,cc) ./ Px;
    end
end
