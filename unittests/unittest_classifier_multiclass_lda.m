% Classifier unit test
%
% Classifier: multiclass_lda

rng(42)
tol = 10e-10;
mf = mfilename;


%% for two classes, multiclass_lda should give the same result as lda
X = randn(1000,10);
clabel = randi(2, size(X,1),1);

% Get classifier params for multiclass LDA and binary LDA
param = mv_get_hyperparameter('multiclass_lda');
param.reg     = 'ridge';
param.lambda  = 0.1;

param_binary = mv_get_hyperparameter('lda');
param_binary.reg     = param.reg;
param_binary.lambda  = param.lambda;

% Train and test classifier
cf_binary = train_lda(param_binary, X, clabel);
cf = train_multiclass_lda(param, X, clabel);

p = corr(cf_binary.w, cf.W);

print_unittest_result('correlate weight vectors for multiclass and binary lda',1, abs(p), tol);


%% for multiple classes, the W's should be orthogonal wrt the within-class scatter matrix
% (though this holds for the empirical covariance only if lambda = 0)
nsamples = 120;
nfeatures = 10;
nclasses = 5;
prop = [];
scale = 0.0001;
do_plot = 0;

[X,clabel] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

param = mv_get_hyperparameter('multiclass_lda');
param.reg     = 'ridge';
param.lambda  = 0;
cf = train_multiclass_lda(param, X, clabel);

% Calculate within-class scatter matrix
nc = arrayfun(@(c) sum(clabel == c), 1:nclasses);
Sw = zeros(nfeatures);
for c=1:nclasses
    Sw = Sw + (nc(c)-1) * cov(X(clabel==c,:));
end

actual = (cf.W' * Sw * cf.W)/ (nsamples-1);
expect = eye(nclasses-1);

print_unittest_result('W''*Sw*W - I = 0',0, norm(actual-expect), tol);
