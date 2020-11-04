function cf = train_naive_bayes(param, X, clabel)
% Trains a Gaussian Naive Bayes classifier.
%
% Usage:
% cf = train_naive_bayes(param, X, clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% param          - struct with hyperparameters:
% .prior         - prior class probability, either 'equal' in which case
%                  every class has an equal probability. Otherwise, provide
%                  a vector of prior probabilities for each class (should
%                  add up to 1). (default 'equal')
%
% IMPLEMENTATION DETAILS:
% The main (and quite naive) assumption in Naive Bayes is that the features
% are conditionally independent. This mean that the posterior probability
% p(C | x1, x2, ..., xn) for class C and features x1, x2, ... xn, is
% proportional to p(C) p(x1|C) p(x2|C) ... p(xn|C).
% In this implementation, probabilities are modelled using Gaussians. That
% is, the class-conditional means and variances are estimated for
% every feature for training.
% At testing time, the maximum a posteriori (MAP) rule is applied to assign
% a sample to the class with the maximum posterior probability.

% (c) Matthias Treder

nclasses = max(clabel);
siz      = size(X);
nfeatures = siz(2:end);

% Indices of class samples
ix = arrayfun(@(c) find(clabel == c), 1:nclasses, 'Un', 0);

%% Estimate class-conditional means and standard deviations
class_means = zeros([nclasses, nfeatures]);
va = zeros([nclasses, nfeatures]);

for cc=1:nclasses
    class_means(cc,:) = mean(X(ix{cc}, :), 1);
    va(cc,:) = var(X(ix{cc}, :), [], 1);
end

%% Set up classifier struct
cf              = [];
cf.class_means  = class_means;
cf.var          = va;
cf.nclasses     = nclasses;

if isfield(param, 'neighbours')
    % pass the neighbours on so that the test function can use it.
    cf.neighbours = param.neighbours;
end

if ischar(param.prior)
    cf.prior        = log(ones(1,nclasses)/nclasses);
else
    % we actually need the log
    cf.prior        = log(param.prior(:)');
end
