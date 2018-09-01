function [clabel,dval] = test_multiclass_lda(cf,X)
% Applies a multiclass LDA classifier to test data and produces class 
% labels.
% 
% Usage:
% clabel = test_multiclass_lda(cf,X)
% 
%Parameters:
% cf             - struct describing the classifier obtained from training 
%                  data. Must contain the field W, see train_multiclass_lda
% X              - [number of samples x number of features] matrix of
%                  test samples
%
%Output:
% clabel        - predicted class labels (1's, 2's, 3's etc)
% dval          - discriminant scores

dval = X * cf.W;

% Calculate Euclidean distance of each sample to each class centroid
dist = arrayfun( @(c) sum( bsxfun(@minus, dval, cf.centroid(c,:)).^2, 2), 1:cf.nclasses, 'Un',0);
dist = cat(2, dist{:});

% For each sample, find the closest centroid and assign it to the
% respective class
clabel = zeros(size(X,1),1);
for ii=1:size(X,1)
    [~, clabel(ii)] = min(dist(ii,:));
end

% % DEBUG - plot data on first two discriminant coordinates
% close all
% for c=1:cf.nclasses 
%     plot(cf.centroid(c,1),cf.centroid(c,2),'+','MarkerSize',18)
%     hold all
% end
% plot(dval(:,1), dval(:,2), 'o'), 
% legend(arrayfun( @(ii) sprintf('Class %d',ii), 1:cf.nclasses,'Un',0))
