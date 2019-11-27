function clabel = test_kernel_fda(cf,X)
% Applies a kernel FDA to test data and produces class labels.
% 
% Usage:
% clabel = test_kernel_fda(cf,X)
% 
%Parameters:
% cf             - classifier. See train_kernel_fda
% X              - [samples x features] matrix of test data
%
%Output:
% clabel     - predicted class labels

if strcmp(cf.kernel,'precomputed')
    % If kernel matrix between test samples and training samples is provided
    % already, it does not need to be calculated
    y = X * cf.A;
else
    
    % Evaluate kernel for test samples using the kernel function
    y = cf.kernelfun(cf, X, cf.Xtrain) * cf.A;
end

% Calculate Euclidean distance of each sample to each class centroid
dist = arrayfun( @(c) sum( bsxfun(@minus, y, cf.class_means(c,:)).^2, 2), 1:cf.nclasses, 'Un',0);
dist = cat(2, dist{:});

% For each sample, find the closest centroid and assign it to the
% respective class
clabel = zeros(size(X,1),1);
for ii=1:size(X,1)
    [~, clabel(ii)] = min(dist(ii,:));
end
