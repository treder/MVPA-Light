% Classifier unit test
%
% Classifier: naive_bayes

tol = 10e-10;
mf = mfilename;

%% test whether probabilities within [0, 1]

%%% Create Gaussian data
nsamples = 60;
nfeatures = 10;
nclasses = 3;
prop = [];
scale = 0.01;
do_plot = 0;

[X_gauss, clabel_gauss] = simulate_gaussian_data(nsamples, nfeatures, nclasses, prop, scale, do_plot);

% get classifier hyperparameter
param = mv_get_hyperparameter('naive_bayes');

% train and test Naive Bayes classifier
cf = train_naive_bayes(param, X_gauss, clabel_gauss);

[pred, dval, prob] = test_naive_bayes(cf, X_gauss);

% Are all returned values between 0 and 1?
print_unittest_result('[prob] all between 0 and 1',1, all(all((prob >= 0) & (prob <= 1))), tol);

%% check probabilities again [using a different prior]
param.prior = [0.1, 0.9, 0.1];

cf = train_naive_bayes(param, X_gauss, clabel_gauss);
[pred, dval, prob] = test_naive_bayes(cf, X_gauss);

% Are all returned values between 0 and 1?
print_unittest_result('[prob with unequal prior] all between 0 and 1',1, all(all((prob >= 0) & (prob <= 1))), tol);
