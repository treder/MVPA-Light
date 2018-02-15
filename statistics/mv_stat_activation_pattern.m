function pattern = mv_stat_activation_pattern(cf, X, clabel)
% Calculates the activation pattern (aka spatial pattern) for linear
% classifiers (lda, logreg, svm with linear kernel). 
%
% Classifier weights (aka spatial filters) are not always interpretable. 
% For instance, the classifier can put a large weight on a feature that 
% measures the noise but has not class-discriminative information. In 
% contrast, the activation pattern is neurophysiologically interpretable. 
% It is related to the concept of structure coefficients used in linear 
% regression.
%
% See Blankertz et al. (2011) and Haufe et al. (2014) for a discussion of
% spatial patterns and spatial filters.
%
% Usage:
% stat = mv_statistics(cfg, result)
%
%Parameters:
% cf             - struct describing the trained classifier. Must contain
%                  field cf.w
% X              - [samples x features] matrix of samples that were used
%                  for training the classifier
% clabel         - [samples x 1] vector of corresponding class labels 
%
% References:
% Blankertz, B., Lemm, S., Treder, M., Haufe, S., & Müller, K. R. (2011). 
% Single-trial analysis and classification of ERP components - A tutorial. 
% NeuroImage, 56(2), 814–825.
%
% Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J. D., 
% Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight 
% vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 
% 96–110. https://doi.org/10.1016/j.neuroimage.2013.10.067

% (c) Matthias Treder 2018

if ~ismatrix(X)
    error('Data X needs to be 2-dimensional')
end

[N,P] = size(X);

mv_check_clabel(clabel);
nclasses = max(clabel);

idx = cell(nclasses,1);
N_per_class = zeros(nclasses,1);

% Find:
% - indices for samples of each class  [idx]
% - number of samples per class [N_per_class]
for cc=1:nclasses
    idx{cc}= (clabel==cc);  % logical indices for samples in class cc
    N_per_class(cc) = sum(idx{cc});
end

% Pool weighted covariance matrix from each class
C = zeros(P);
for cc=1:nclasses
    C = C + cov(X(idx{cc},:),1) * N_per_class(cc);
end
C = C / N;

% Activation pattern
pattern = C * cf.w;

