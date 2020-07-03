% Classifier unit test
%
% Classifier: logreg

rng(42)
tol = 10e-10;
mf = mfilename;

% Random data
N = 500;
nfeatures = 100;
X = randn(N, nfeatures);
X(2:2:end,:) = X(2:2:end,:) + 0.5;
clabel = ones(N, 1); 
clabel(2:2:end) = 2;

%% check "prob" parameter: all returned values must be probabilities

% Get classifier params
param = mv_get_hyperparameter('logreg');
param.prob = 1;

% Train and test classifier
cf = train_logreg(param, X, clabel);
[~,~,prob] = test_logreg(cf,X);

% all probabilities <=1
print_unittest_result('check prob parameter: all 0 <= prob <= 1',1, all(prob>=0 & prob<=1), tol);

%% check bias for unbalanced data with l2 reg (without bias correction): if there's more class 1 the average dval should be positive and vice versa

% Create two unbalanced class labels
clabel_unbalanced1 = clabel;
clabel_unbalanced2 = clabel;
clabel_unbalanced1(1:N-10) = 1;
clabel_unbalanced2(1:N-10) = 2;

% Train and test classifier
param = mv_get_hyperparameter('logreg');
param.correct_bias = 0;
param.reg = 'l2';
param.lambda = 1;

% mean dvals should be positive for both classes (positive shift due to bias towards positive class)
cf = train_logreg(param, X, clabel_unbalanced1);
[~,dval] = test_logreg(cf,X);
dval_unbalanced1 = [mean(dval(clabel_unbalanced1==1)), mean(dval(clabel_unbalanced1==2))];

print_unittest_result('check dval>0 for both classes in unbalanced data (more class 1 data) with l2 reg',1, all(dval_unbalanced1 > 0),  tol);

% mean dvals should be negative for both classes (negative shift due to bias towards negative class)
cf = train_logreg(param, X, clabel_unbalanced2);
[~,dval] = test_logreg(cf,X);
dval_unbalanced2 = [mean(dval(clabel_unbalanced2==1)), mean(dval(clabel_unbalanced2==2))];

print_unittest_result('check dval>0 for both classes in unbalanced data (more class 2 data) with l2 reg',1, all(dval_unbalanced2 < 0),  tol);

%% check bias correction for unbalanced data with l2 reg: with bias correction positive samples should give a mean positive dval, and vice versa for negative

% Train and test classifier
param = mv_get_hyperparameter('logreg');
param.correct_bias = 1;
param.reg = 'l2';
param.lambda = 1;

% mean dvals should be positive for both classes (positive shift due to bias towards positive class)
cf = train_logreg(param, X, clabel_unbalanced1);
[~,dval] = test_logreg(cf,X);
dval_unbalanced1 = [mean(dval(clabel_unbalanced1==1)), mean(dval(clabel_unbalanced1==2))];

print_unittest_result('check with bias correction (dval class 1 > 0, dval class 2 < 0) with l2 reg',1, (dval_unbalanced1(1)>0 && dval_unbalanced1(2)<0),  tol);

% mean dvals should be negative for both classes (negative shift due to bias towards negative class)
cf = train_logreg(param, X, clabel_unbalanced2);
[~,dval] = test_logreg(cf,X);
dval_unbalanced2 = [mean(dval(clabel_unbalanced2==1)), mean(dval(clabel_unbalanced2==2))];

print_unittest_result('check with bias correction (dval class 1 > 0, dval class 2 < 0) with l2 reg',1, (dval_unbalanced2(1)>0 && dval_unbalanced2(2)<0),  tol);
