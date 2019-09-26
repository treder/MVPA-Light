function [clabel, dval, prob, coordinates] = test_multiclass_lda(cf,X)
% Applies a multiclass LDA classifier to test data and produces class 
% labels.
% 
% Usage:
% clabel = test_multiclass_lda(cf,X)
% 
%Parameters:
% cf             - struct describing the classifier obtained from training 
%                  data. Must contain the field W, see train_multiclass_lda
% X              - [samples x features] matrix of test samples
%
%Output:
% clabel        - [samples x 1] vector of predicted class labels (1's, 2's, 3's etc)
% dval          - [samples x classes] matrix of decision values (distances to class centroids)
% prob          - [samples x classes] matrix of posterior class probabilities
% coordinates   - [samples x (classes-1)] matrix of discriminant
%                 coordinates obtained after projecting the data into the
%                 discriminant subspace

% discriminant coordinates (data projected onto subspace)
coordinates = X * cf.W;

% Calculate Euclidean distance of each sample to each class centroid
dval = arrayfun( @(c) sum( bsxfun(@minus, coordinates, cf.centroid(c,:)).^2, 2), 1:cf.nclasses, 'Un',0);
dval = cat(2, dval{:});

% For each sample, find the closest centroid and assign it to the
% respective class
clabel = zeros(size(X,1),1);
for ii=1:size(X,1)
    [~, clabel(ii)] = min(dval(ii,:));
end

if nargout > 2
    % To obtain posterior probabilities, we evaluate a multivariate normal
    % pdf at the test data point. Since W diagonalises and whitens the
    % space each class is distributed as N(?,1) where ? is the class
    % centroid. Since the class centroids have already been subtracted in
    % dval, we can simply evaluate the standard normal distribution N(0,1).
    
    % these are the likelihoods P(x|c)
    prob = 1/sqrt(2*pi) * exp(-(dval.^2)/2);
    
    % normalize (assuming equal priors) to get the posterior class
    % probabilities P(c|x)
    prob = prob ./ repmat(sum(prob,2), [1, cf.nclasses]);
end

% % DEBUG - plot data on first two discriminant coordinates
% close all
% for c=1:cf.nclasses 
%     plot(cf.centroid(c,1),cf.centroid(c,2),'+','MarkerSize',18)
%     hold all
% end
% plot(dval(:,1), dval(:,2), 'o'), 
% legend(arrayfun( @(ii) sprintf('Class %d',ii), 1:cf.nclasses,'Un',0))
