% Classifier unit test
%
% Classifier: lda

rng(42)
tol = 10e-8;
mf = mfilename;

% Random data
nfeatures = 100;
X = randn(1000, nfeatures);
clabel = randi(2, size(X,1),1);

%% check "scale" parameter: if scale = 1, training data should be scaled such that mean(class1)=1 and mean(class2)=-1

% Get classifier params
param = mv_get_hyperparameter('lda');
param.scale = 1;

% Train and test classifier
cf = train_lda(param, X, clabel);
[~,dval] = test_lda(cf,X);

% Does class 1 project to +1 ?
print_unittest_result('check scale parameter for class 1',1, mean(dval(clabel==1)), tol);

% Does class 2 project to -1 ?
print_unittest_result('check scale parameter for class 2',-1, mean(dval(clabel==2)), tol);

%% check "prob" parameter: if prob = 1, probabilities should be returned as third parameter

% Get classifier params
param = mv_get_hyperparameter('lda');
param.prob = 1;

% Train and test classifier
cf = train_lda(param, X, clabel);
[~,~,prob] = test_lda(cf,X);

% Are all returned values between 0 and 1?
print_unittest_result('check prob parameter',1, all(abs(prob)<=1),  tol);

%% check "lambda" parameter: if lambda = 1, w should be collinear with the difference between the class means

% Get classifier params
param = mv_get_hyperparameter('lda');
param.reg       = 'shrink';
param.lambda    = 1;

cf = train_lda(param, X, clabel);

% Difference between class means
m = mean(X(clabel==1,:)) - mean(X(clabel==2,:));

% Correlation between m and cf.w
p = corr(m', cf.w);

% Are all returned values between 0 and 1?
print_unittest_result('check w parameter for lambda=1 (equal to diff of class means?)',1, p,  tol);

%% Equivalence between ridge and shrinkage regularisation

% Get classifier param for shrinkage regularisation
param_shrink = mv_get_hyperparameter('lda');
param_shrink.reg   = 'shrink';
param_shrink.lambda = 0.5;

% Determine within-class scatter matrix (we need its trace)
Sw= sum(clabel==1) * cov(X(clabel==1,:),1) + sum(clabel==2) * cov(X(clabel==2,:),1);

% Determine the equivalent ridge parameter using the formula
% ridge = shrink/(1-shrink) * trace(C)/P
% Obviously the formula only works for shrink < 1
param_ridge = param_shrink;
param_ridge.reg      = 'ridge';
param_ridge.lambda   = param_shrink.lambda/(1-param_shrink.lambda) * trace(Sw)/nfeatures;

% Train classifiers with both types of regularisation
cf_shrink = train_lda(param_shrink, X, clabel);
cf_ridge = train_lda(param_ridge, X, clabel);

p = corr(cf_ridge.w, cf_shrink.w);

print_unittest_result('Corr between ridge and shrinkage classifier weights',1, p, tol);

%% Ridge regularization: establish equivalence between primal and dual form
clear cf_ridge cf_shrink param_ridge param_shrink
[X, clabel] = simulate_gaussian_data(1000, 100, 2, [], [], 0);
% X = zscore(X,[], 1);
X_train = X;
X_train([1:100, end-99:end],:) = [];
X_test = X([1:100, end-99:end],:);
clabel_train = clabel; 
clabel_train([1:100, end-99:end]) = [];
clabel_test = clabel([1:100, end-99:end]);

% P >> N
X1 = rand(10, 5000) + 2;  
clabel1 = [ones(5,1); 2*ones(5,1)];
X1 = zscore(X1, [], 1);

% N >> P
X2 = rand(1000, 10) + 2;  
clabel2 = [ones(500,1); 2*ones(500,1)];

param = mv_get_hyperparameter('lda');
param.reg = 'ridge';
param.lambda = 0.5;

param_primal = param;
param_dual   = param;
param_primal.form       = 'primal';
param_dual.form         = 'dual';

% Gaussian data
for lambda = [0.1, 1, 10, 100]
    param_primal.lambda = lambda;
    param_dual.lambda = lambda;
    % Train classifiers with primal/dual form
    cf_primal = train_lda(param_primal, X_train, clabel_train);
    cf_dual = train_lda(param_dual, X_train, clabel_train);
    c_primal = test_lda(cf_primal, X_test);
    c_dual = test_lda(cf_dual, X_test);
    
    print_unittest_result(sprintf('[ridge, gaussian] primal vs dual form for lambda=%2.2f',lambda), cf_primal.w, cf_dual.w, tol);
    print_unittest_result(sprintf('[ridge, gaussian] compare testlabels for lambda=%2.4f',lambda), c_primal, c_dual, tol);
end

% P>>N data
for lambda = [0.0001, 0.1, 1, 10, 100]
    param_primal.lambda = lambda;
    param_dual.lambda = lambda;
    % Train classifiers with primal/dual form
    cf_primal = train_lda(param_primal, X1, clabel1);
    cf_dual = train_lda(param_dual, X1, clabel1);
    c_primal = test_lda(cf_primal, X1);
    c_dual = test_lda(cf_dual, X1);
    
    print_unittest_result(sprintf('[ridge, P>>N] corr(w_primal,w_dual) for lambda=%2.4f',lambda), 1, corr(cf_primal.w, cf_dual.w), tol);
    print_unittest_result(sprintf('[ridge, P>>N] b: primal vs dual form for lambda=%2.4f',lambda), cf_primal.b, cf_dual.b, tol);
    print_unittest_result(sprintf('[ridge, P>>N] compare testlabels for lambda=%2.4f',lambda), c_primal, c_dual, tol);
end

% N>>P data
for lambda = [0.1, 1, 10, 100]
    param_primal.lambda = lambda;
    param_dual.lambda = lambda;
    % Train classifiers with primal/dual form
    cf_primal = train_lda(param_primal, X2, clabel2);
    cf_dual = train_lda(param_dual, X2, clabel2);
    c_primal = test_lda(cf_primal, X2);
    c_dual = test_lda(cf_dual, X2);
    
    print_unittest_result(sprintf('[ridge, N>>P] corr(w_primal,w_dual) for lambda=%2.2f',lambda), 1, corr(cf_primal.w, cf_dual.w), tol);
    print_unittest_result(sprintf('[ridge, P>>N] compare testlabels for lambda=%2.4f',lambda), c_primal, c_dual, tol);
end

%% Shrinkage regularization: establish equivalence between primal and dual form
param = mv_get_hyperparameter('lda');
param.reg = 'shrink';

param_primal = param;
param_dual   = param;
param_primal.form       = 'primal';
param_dual.form         = 'dual';

% Gaussian data
for lambda = [0.001, 0.01, 0.1, 0.4, 0.9, 1]
    param_primal.lambda = lambda;
    param_dual.lambda = lambda;
    % Train classifiers with primal/dual form
    cf_primal = train_lda(param_primal, X, clabel);
    cf_dual = train_lda(param_dual, X, clabel);
    
%     print_unittest_result(sprintf('[shrinkage, gaussian data] |primal - dual| for lambda=%1.2f',lambda), cf_primal.w, cf_dual.w, tol);
    print_unittest_result(sprintf('[shrinkage, gaussian data] corr(primal,dual) for lambda=%1.2f',lambda), 1, corr(cf_primal.w, cf_dual.w), tol);
end

% P>>N data
for lambda = [0.001, 0.01, 0.1, 0.4, 0.9, 1]
    param_primal.lambda = lambda;
    param_dual.lambda = lambda;
    % Train classifiers with primal/dual form
    cf_primal = train_lda(param_primal, X1, clabel1);
    cf_dual = train_lda(param_dual, X1, clabel1);
    
    print_unittest_result(sprintf('[shrinkage, P>>N data] corr(primal,dual) for lambda=%1.2f',lambda), 1, corr(cf_primal.w, cf_dual.w), tol);
%     print_unittest_result(sprintf('[shrinkage, P>>N data] w: primal vs dual form for lambda=%1.2f',lambda), cf_primal.w, cf_dual.w, tol);
end

% N>>P data
for lambda = [0.001, 0.01, 0.1, 0.4, 0.9, 1]
    param_primal.lambda = lambda;
    param_dual.lambda = lambda;
    % Train classifiers with primal/dual form
    cf_primal = train_lda(param_primal, X2, clabel2);
    cf_dual = train_lda(param_dual, X2, clabel2);
    
    print_unittest_result(sprintf('[shrinkage, P>>N data] corr(primal,dual) for lambda=%1.2f',lambda), 1, corr(cf_primal.w, cf_dual.w), tol);
%     print_unittest_result(sprintf('[shrinkage, N>>P data] primal vs dual form for lambda=%1.2f',lambda), cf_primal.w, cf_dual.w, tol);
end

%% Shrinkage regularization with lambda='auto': establish equivalence between primal and dual form
param = mv_get_hyperparameter('lda');
param.reg = 'shrink';
param.lambda = 'auto';

param_primal = param;
param_dual   = param;
param_primal.form       = 'primal';
param_dual.form         = 'dual';

% Gaussian data
cf_primal = train_lda(param_primal, X, clabel);
cf_dual = train_lda(param_dual, X, clabel);
    
print_unittest_result('[shrinkage, gaussian] lambda: primal=dual for lambda=auto', cf_primal.lambda, cf_dual.lambda, tol);
% print_unittest_result('[shrinkage, gaussian] w: primal=dual form for lambda=auto', cf_primal.w, cf_dual.w, tol);
print_unittest_result('[shrinkage, gaussian] corr(primal,dual) lambda=auto', 1, corr(cf_primal.w, cf_dual.w), tol);

% P>>N data
cf_primal = train_lda(param_primal, X1, clabel1);
cf_dual = train_lda(param_dual, X1, clabel1);

print_unittest_result('[shrinkage, P>>N] lambda: primal=dual for lambda=auto', cf_primal.lambda, cf_dual.lambda, tol);
% print_unittest_result('[shrinkage, P>>N] w: primal=dual form for lambda=auto', cf_primal.w, cf_dual.w, tol);
print_unittest_result('[shrinkage, P>>N] b: primal=dual form for lambda=auto', cf_primal.b, cf_dual.b, tol);
print_unittest_result('[shrinkage, P>>N] corr(primal,dual) lambda=auto', 1, corr(cf_primal.w, cf_dual.w), tol);

% N>>P data
cf_primal = train_lda(param_primal, X2, clabel2);
cf_dual = train_lda(param_dual, X2, clabel2);

print_unittest_result('[shrinkage, N>>P] lambda: primal=dual for lambda=auto', cf_primal.lambda, cf_dual.lambda, tol);
% print_unittest_result('[shrinkage, N>>P] w: primal vs dual form for lambda=auto', cf_primal.w, cf_dual.w, tol);
print_unittest_result('[shrinkage, N>>P] corr(primal,dual) lambda=auto', 1, corr(cf_primal.w, cf_dual.w), tol);
