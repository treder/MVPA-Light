% UNDERSTANDING TRAIN AND TEST FUNCTIONS
%
% The backbone of MVPA Light's high-level functions such as mv_classify and
% mv_regress are the train and test functions of the models. They are found
% in the model subfolder. Many standard MVPA problems can be solved using
% the high-level functions, but for some advanced analyses it is
% necessary to directly access the train and test functions. This scenario
% is examined in this tutorial.
%
% Contents:
% (1) Train and test functions for classification
% (2) Train and test functions for regression
%
% Note: If you are new to working with MVPA-Light, make sure that you
% complete the introductory tutorials first:
% - getting_started_with_classification
% - getting_started_with_regression
% They are found in the same folder as this tutorial.

close all
clear

%% Loading example data
[dat, clabel] = load_example_data('epoched3');
X = dat.trial;

%% (1) Train and test functions for classification

% To see how to call a train function, let us inspect the help of LDA
help train_lda
% train_lda expects three parameters: a parameter struct, the data X and
% class labels clabel. The param struct contains the hyperparameters of the
% classifier. We should first get the default values which we can overwrite
% if necessary. mv_get_hyperparameter provides the default values
param = mv_get_hyperparameter('lda')

% Let us change the lambda hyperparameter from 'auto' to 0.01
param.lambda = 0.01;

% Now, let's train a LDA classifier
lda = train_lda(param, X, clabel);

% This did not work, why?
% Our data is [samples x features x time points], but every train function
% expects a 2D array [samples x features]. We can create this by
% selecting a specific time point eg sample 100 from X.
X = squeeze(X(:,:,100));

% Now the training will work
lda = train_lda(param, X, clabel);

% Let's inspect the model
lda
% It has four parameters, the weight vector w, bias term b, prob
% (hyperparameter specifying whether or not we want to calculate
% probabilities), and the regularization value lambda which we defined
% above

% We can now test the classifier. We do not have a separate test set, so we
% will simply test the classifier on the same dataset. 
predlabel = test_lda(lda, X);

% predlabel is a vector of predicted class labels. Let's look at the first
% 10 predicted labels and compare them to the real class labels
predlabel(1:10)'
clabel(1:10)'

% If we want decision values (dvals) we can specify a second output
% argument
[predlabel, dval] = test_lda(lda, X);

% The first 10 values are negative (corresponding to class 2)
dval(1:10)'

% Look at the distribution of the decision values using a boxplot.
% dvals should be positive. For clabel 1 and negative for clabel 2. 
% dval = 0 is the decision boundary
figure
clf
boxplot(dval, clabel)
hold on, grid on
plot(xlim, [0 0],'k--')
ylabel('Decision values')
xlabel('Class')
title('Decision values per class')

% We can also request probability values. They are given as the third
% output argument
[~, ~, prob] = test_lda(lda, X);

% This did not work! The reason is that in order to produce probabilities,
% we need to estimate multivariate Gaussian distributions on the training
% data. To this end, we need to set the .prob hyperparameter to 1 and train
% again.
param.prob = 1;
lda = train_lda(param, X, clabel);

% Now looking at the model we see there are additional fields used by the
% test function for calculating the probabilities 
lda

% Now we can obtain probabilities
[~, ~, prob] = test_lda(lda, X);

% The interpretation of the probability is "probability that the sample
% belongs to class 1". Looking at the first 10 values 
prob(1:10)'
% they are all below 0.5 except for one instance, which is why most of them
% were assigned to class 2

% Look at the distribution of the probabilities. prob should be higher
% for clabel 1 than clabel 2
figure
clf
boxplot(prob, clabel)
hold on, grid on
plot(xlim, [0.5 0.5],'k--')
ylabel('Probability for class 1')
xlabel('Class')
title('Class probabilities P(class=1) per class')

% We can also calculate a performance metric by hand using the 
% mv_calculate_performance function. 
help mv_calculate_performance

% Let us calculate AUC based on the dvals 
auc = mv_calculate_performance('auc', 'dval', dval, clabel);

%%%%%% EXERCISE 1 %%%%%%
% Repeat the analysis done here using a different classifier and a
% different performance metric. Train a Logistic Regression model (called
% logreg in MVPA Light). Set the regularization to l2 and and lambda to 0.001.
% Obtain the predicted labels on X, then calculate the F1 score.
%%%%%%%%%%%%%%%%%%%%%%%%


%% (2) Train and test functions for regression
% Regression train/test functions have already been introduced in the last
% section of getting_started_with_regression. Here, we will delve somewhat
% deeper into the hyperparameter and the outputs. The same data will be
% used.

% We use the simulated ERP dataset introduced in the first section of 
% getting_started_with_regression. Refer to the description there for more
% information.
n_trials = 300;
time = linspace(-0.2, 1, 201);
n_time_points = numel(time);
pos = 100;
width = 10;
amplitude = 3*randn(n_trials,1) + 3;
weight = abs(randn(64, 1));
scale = 1; 
X_erp = simulate_erp_peak(n_trials, n_time_points, pos, width, amplitude, weight, [], scale);
y = amplitude + 0.5 * randn(n_trials, 1);

% Regression train/test functions work very similarly to their
% classification counterparts: Every train function expects three
% parameters: a parameter struct, the data X and a vector of responses
% serving as targets. The param struct contains the hyperparameters of the
% model. We should first get the default values which we can overwrite
% if necessary. mv_get_hyperparameter provides the default values. Let's
% start with Ridge regression:
param = mv_get_hyperparameter('ridge')

% To understand the meaning of the hyperparameters, refer to the help
help train_ridge

% Let's train the model
model = train_ridge(param, X_erp, y);
% This did not work, why? Our data is [samples x features x time points], 
% but every train function always expect a 2D array [samples x features]. 
% We can create this by selecting a specific time point eg sample 100 from X.
X100 = squeeze(X_erp(:,:,100));

% Let's try again
model = train_ridge(param, X100, y)
% The model contains 4 values, the two most important ones being the vector
% of regression coefficients w and the intercept b

%%%%%% EXERCISE 2 %%%%%%
% We want to investigate the effect of lambda on the magnitude of the
% regression coefficients. It is said that lambda has a shrinkage effect
% i.e. the magnitude of the regression coefficients w decreases with
% increasing lambda. To check this, train multiple Ridge models with lambda
% ranging from 0 to 10 in steps of 1. Calculate the norm of w for every
% value of lambda. Create a plot with lambda on the x-axis and the norm on
% the y-axis.
% Hint: use a for loop and the norm() function.
%%%%%%%%%%%%%%%%%%%%%%%%

% Call the test function to get the predicted responses on the training
% set.
ypred = test_ridge(model, X100);

% displaying the true targets and the predictions side by side we see that
% there is a good correspondence
[y, ypred]

% We can now calculate performance measures by hand by using the
% mv_calculate_performance function
R_squared_value = mv_calculate_performance('r_squared', [], ypred, y)

%%%%%% EXERCISE 3 %%%%%%
% Use ypred and y to calculate MAE and MSE.
%%%%%%%%%%%%%%%%%%%%%%%%

% Congrats, you finished the tutorial!

%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
help train_logreg

% Get Logistic Regression hyperparameters and change the regularization
param = mv_get_hyperparameter('logreg');
param.reg       = 'l2';
param.lambda    = .001;

% Train the model
logreg = train_logreg(param, X, clabel);

% The model has weights w and bias term b as main parameters
logreg

% Get test predictions
predlabel = test_logreg(logreg, X);

% Calculate F1 score based on the predicted class labels
f1 = mv_calculate_performance('f1', 'clabel', predlabel, clabel);
f1

%% SOLUTION TO EXERCISE 2
% We can create a vector lambdas that contains all desired lambda values
% and then run a loop over these lambdas. In each iteration of the loop, we
% train a model with the current value of lambda and record the norm of w
% in a vector called norms.
param = mv_get_hyperparameter('ridge');
lambdas = 1:10;
norms = zeros(numel(lambdas), 1);

for ix = 1:numel(lambdas)
    param.lambda = lambdas(ix);
    model = train_ridge(param, X100, y);
    norms(ix) = norm(model.w);
end

figure
plot(lambdas, norms, 'ro-', 'MarkerSize', 12, 'MarkerFaceColor', 'w')
xlabel('Lambda'), ylabel('Norm of w')
grid on
% The figure indeed shows that the norm of w decreases as lambda increases,
% showcasing the shrinkage effect of lambda.

%% SOLUTION TO EXERCISE 3
mae = mv_calculate_performance('mae', [], ypred, y)
mse = mv_calculate_performance('mse', [], ypred, y)
