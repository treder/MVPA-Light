% GETTING STARTED WITH REGRESSION
% 
% This is the go-to tutorial if you are new to the toolbox and want to
% get started with regression. It covers the following topics:
%
% (1) Loading example data
% (2) Regression with cross-validation and explanation of the cfg struct
% (3) Plotting results
% (4) Compare Ridge, Kernel Ridge, and Support Vector Regression
% (5) Transfer learning (cross regression)
%
% It is recommended that you work through this tutorial step by step. To
% this end, copy the lines of code that you are currently reading and paste
% them into the Matlab console. 
%
% There will be exercises throughout the tutorial. Try to do the exercise,
% you can then check your code against the solution at the end of this
% file.
%
% Note: we will use machine learning terminology such as cross-validation,
% features, classifier, train and test set. If you are unfamiliar with
% these terms, please check out the glossary in section 1.1 of the
% MVPA-Light paper (www.frontiersin.org/articles/10.3389/fnins.2020.00289)
% and read the tutorial papers that are mentioned there.
%
% Troubleshooting: If the code crashes on your computer, make sure that you
% always have the latest version of the toolbox and that you are using
% Matlab version 2012 or newer.
%
% Documentation:
% The Github Readme file is the most up-to-date documentation of the
% toolbox. You will find an explanation of the functions, models,
% metrics and parameters there: github.com/treder/MVPA-Light/blob/master/README.md
%
% Next steps: Once you finished this tutorial, you can continue with one
% of the other tutorials:
% - advanced_classification
% - understanding_metrics
% - understanding_preprocessing
% - understanding_statistics
% - understanding_train_and_test_functions
%
% You can also check out the Github repo https://github.com/treder/MVPA-Light-Paper
% It contains all the analysis scripts used in the MVPA-Light paper.

close all
clear

%% (1) Simulate ERP data for regression

% MVPA-Light does not come with a EEG dataset prepared for regression.
% Therefore, we will use the simulate_erp_peak function to create an
% artificial dataset. We simulate a event-related EEG dataset that has a 
% ERP peak whose amplitude varies from trial to trial. We will then
% create a target variable y which is serves as a proxy to this amplitude
% and will be used as a target in our regression problem.

% Copy-paste the following code into the console. Then copy-paste the
% code below that produces the visualization.
n_trials = 300;
time = linspace(-0.2, 1, 201);
n_time_points = numel(time);
pos = 100;          % position of ERP peak in samples
width = 10;         % width of ERP peak in samples
amplitude = 3*randn(n_trials,1) + 3; % amplitudes of ERP peak (different in each trial)
weight = abs(randn(64, 1)); % projection weight from ERP signal to 64 electrodes 
scale = 1; % scale of the noise

X = simulate_erp_peak(n_trials, n_time_points, pos, width, amplitude, weight, [], scale);

% Plot some single trials (different lines represent different electrodes)
figure
subplot(1,3,1), plot(time, squeeze(X(1,:,:))'), title('Trial 1 (all channels)')
subplot(1,3,2), plot(time, squeeze(X(11,:,:))'), title('Trial 11 (all channels)')
subplot(1,3,3), plot(time, squeeze(X(200,:,:))'), title('Trial 200 (all channels)')

% Calculate the average ERP (different lines represent different electrodes)
figure
plot(time, squeeze(mean(X,1))')
title('Average (all channels)')
xlabel('Time'), ylabel('Amplitude')

% Now it's time to create the target variable y. In a real experiment, y 
% could be reaction time or trial number. Since our data is artificial, we 
% create y from the amplitude. In particular, we create y as a noisy
% version of the ERP amplitude by adding some Gaussian noise:
y = amplitude + 0.5 * randn(n_trials, 1);

%% (2) Regression with cross-validation and explanation of the cfg struct

% The main function for performing regression is mv_regress. 
% Let's jump straight into it:
cfg = [];
perf = mv_regress(cfg, X, y);

% There seems to be a lot going on here, so let's unpack the questions that
% might come up:
% 1. What happened? If we read the output on the console, we can figure out
% the following: mv_regress performed a cross-validation regression
% analysis using 5-fold cross-validation (k=5), 5 repetitions, using an 
% RIDGE regression model. This is simply the default behaviour if we don't 
% specify anything else.

% 2. What is perf? Perf refers to 'performance metric', a measure of how
% good of a job the model did. By default, it calculates the mean absolute
% error (or MAE for short). Since the regression is performed for each time
% point, we get a vector of MAEs (one for each of the 131 time points). The
% MAE has been cross-validated, that is, separate partitions of the data
% have been used for training and testing.
close all
plot(time, perf)
xlabel('Time'), ylabel('MAE')
% We can see that the MAE is lowest at the time of the ERP peak. This makes
% sense because this is when the variable y is clearly related to the data.

% 3. What does cfg do, it was empty after all?
% cfg controls all aspects of the analysis: choosing the
% model, a metric, preprocessing and defining the cross-validation. If
% it is unspecified, it is simply filled with default values. 

%%%%%% EXERCISE 1 %%%%%%
% The regression model can be specified by setting cfg.model = ... 
% Look at the available models at 
% https://github.com/treder/MVPA-Light#regression-models-
% Run the analysis again, this time using a Kernel Ridge model with a
% polynomial kernel.
%%%%%%%%%%%%%%%%%%%%%%%%

% Now we know how to set a model, let's see how we can change the
% metric that we want to be calculated.  Let's go for R-squared 
% metric instead of MAE. R-squared represents the proportional variance
% explained by the model. Note that in cross-validated data R-squared can
% be negative.
% A list of metrics is available at https://github.com/treder/MVPA-Light#regression-performance-metrics
cfg             = [];
cfg.metric      = 'r_squared';
perf = mv_regress(cfg, X, y);

% We can also calculate both R-squared and MAE at the same time using a cell
% array. Now perf will be a cell array. The first cell is the R-squared values,
% the second cell is MAEs. 
cfg = [];
cfg.metric      = {'r_squared', 'mae'};
perf = mv_regress(cfg, X, y);

perf

%%%%%% EXERCISE 2 %%%%%%
% Look at the available classification metrics at 
% https://github.com/treder/MVPA-Light#regression-performance-metrics
% Run the analysis again, this time calculating mean squared error.
%%%%%%%%%%%%%%%%%%%%%%%%

% We know now how to define the classifier and the performance metric. We
% still need to understand how to change the cross-validation scheme. Let us
% perform k-fold cross-validation with 10 folds (i.e. 10-fold
% cross-validation) and 2 repetitions. Note how the output on the console 
% changes.
cfg = [];
cfg.k           = 10;
cfg.repeat      = 2;
perf = mv_regress(cfg, X, y);

%%%%%% EXERCISE 3 %%%%%%
% Look at the description of cross-validation at 
% https://github.com/treder/MVPA-Light/blob/master/README.md#cv
% Do the classification again, but instead of k-fold cross-validation use
% a holdout set and designate 20% of the data for testing.
%%%%%%%%%%%%%%%%%%%%%%%%


%% (3) Plotting results
% So far, we have plotted the results by hand using Matlab's plot
% function. For a quick and dirty visualization, MVPA-Light has a function
% called mv_plot_result. It plots the results and nicely lays out the axes
% for us. To be able to use it, we need the result struct, which is simply
% the second output argument of mv_regress.
cfg             = [];
[perf, result] = mv_regress(cfg, X, y);

% Now call mv_plot_result  passing result as an input argument. We will obtain 
% line plot representing MAE across time.
% The shaded area represents the standard deviation across folds and
% repetitions, an heuristic marker of how variable the performance measure
% is for different test sets.
close all
mv_plot_result(result)

% We can name the dimensions of X. The dimension names will be added to the
% output, and the x-axis will be labeled as time points when we plot the
% result again
cfg             = [];
cfg.dimension_names = {'samples' 'channels', 'time points'}; 
[~, result] = mv_regress(cfg, X, y);

mv_plot_result(result)

% The x-axis depicts the sample number, not the actual time points. To get the
% x-axis in seconds, we can provide the time points as an extra argument to
% the function call. 
mv_plot_result(result, time)

%%%%%% EXERCISE 4 %%%%%%
% Calculate MSE, MAE, and R-squared at once and plot the result using
% mv_plot_result.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (4) Compare Ridge, Kernel Ridge, and Support Vector Regression
% To illustrate how kernels tackle non-linear problems, we will
% here create an 1-dimensional non-linear dataset. We will then train ridge
% regression, kernel ridge, and Support Vector Regression (SVR) models and
% compare them. 
% Note: The SVR model requires an installation of LIBSVM, see
% train_libsvm.m for details

% We simulate sinusoidal data with a quadratic trend
x = linspace(0, 12, 100)';
y = -.1 * x.^2 + 3 * sin(x);     % SINUSOID WITH QUADRATIC TREND
y_plus_noise  = y + randn(length(y), 1);

% plot simulated data
close all
plot(x,y, 'r', 'LineWidth', 2)
hold on
plot(x,y_plus_noise, 'ko')
legend({'True signal' 'Data (signal plus noise)'})
title('True signal and data')

% We want to plot the predictions of the models, not some summary
% statistic such as MAE. To this end, we perform training and testing by
% hand:
% Ridge Regression: Train model and get predicted values
param_ridge = mv_get_hyperparameter('ridge');       % get default hyperparameters
model = train_ridge(param_ridge, x, y_plus_noise);  % train the model
y_ridge = test_ridge(model, x);                     % obtain the predictions

% Kernel Ridge: Train model and get predicted values
param_krr = mv_get_hyperparameter('kernel_ridge');
model = train_kernel_ridge(param_krr, x, y_plus_noise);
y_kernel_ridge = test_kernel_ridge(model, x);

% SVR: Train model and get predicted values
% We will use the LIBSVM toolbox here, which supports both 
% classification (SVM) and regression (SVR).
param_svr = mv_get_hyperparameter('libsvm');
% Set svm_type to 3 for support vector regression
param_svr.svm_type = 3; 
model = train_libsvm(param_svr, x, y_plus_noise);
y_svr = test_libsvm(model, x);

% Plot the data and the predictions of the three models in a plot
figure,hold on
plot(x,y_plus_noise, 'ko')
plot(x, y_ridge, 'b')   % ridge prediction
plot(x, y_kernel_ridge, 'k')   % kernel ridge prediction
plot(x, y_svr, 'g')   % SVR prediction

legend({'Data' 'Ridge regression' 'Kernel ridge' 'SVR'})
title('Predictions')
% We can see that the Ridge models underfits the data. It produces a
% straight line that cannot fully model the complexity of the data. In
% contrast, the Kernel Ridge and SVR models nicely model the nonlinear
% data. Their predictions are close to the true underlying function.

%%%%%% EXERCISE 5 %%%%%%
% Perform the same analysis using data that represents a sawtooth function.
% The formula is given by 
% y = 2*mod(x, 3) + 0.4 * x; % SAWTOOTH FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%

%% (5) Transfer regression (cross regression)
% In transfer regression (or cross regression), one dataset is used for 
% training, another one for testing. Technically, this is the same as in 
% cross-validation, where one fold is defined for training and a separate 
% fold for testing. However, in transfer regression we assume that the two 
% datasets come from potentially different distributions (e.g. two different
% participants, two different phases in an experiment etc). 
% Transfer regression is implemented in mv_regress.
% Let's first recreate the dataset from example 1.
X = simulate_erp_peak(n_trials, n_time_points, pos, width, amplitude, weight, [], scale);
y = amplitude + 0.5 * randn(n_trials, 1);

% Now create a second ERP dataset. It has the same parameters as
% the first one, but 10 more trials.
n_trials2 = n_trials + 10;
amplitude2 = 3*randn(n_trials2,1) + 3;
X2 = simulate_erp_peak(n_trials2, n_time_points, pos, width, amplitude2, weight, [], scale);
y2 = amplitude2 + 0.5 * randn(n_trials2, 1);

% Now let's perform the cross regression. We only need to add X2 and y2 as
% extra inputs to mv_regress.
cfg = [];
cfg.model           = 'ridge';
cfg.dimension_names = {'samples' 'channels' 'time points'};
[perf, result] = mv_regress(cfg, X, y, X2, y2);

% The model has been trained for each time point in dataset 1 and tested at
% the same time point in dataset 2 
mv_plot_result(result)

% Now let's create a dataset with fewer time points
n_time_points2 = 25;
X2 = simulate_erp_peak(n_trials2, n_time_points2, pos, width, amplitude2, weight, [], scale);
% this leads to an ERROR so let's wrap it into a try-except statement
try
    [perf, result] = mv_regress(cfg, X, y, X2, y2);
catch err
    fprintf('[ERROR] Call to mv_regress failed:\n%s\n', err.message)
end

% However, we can fix this by defining time as a generalization
% dimension: a model is trained / tested for every combination of
% train/test time, and it does not matter any more whether they match in length.
cfg.generalization_dimension = 3;
[perf, result] = mv_regress(cfg,  X, y, X2, y2);

% The dimensions of the result are [time points x train frequencies x test
% frequences]. The generalization dimension is always moved to the end,
% this is why time points come first. mv_plot_result does not plot 3D data,
% so we leave it at just looking at the dimensions here.
size(perf)


%% The End
% Congrats, you made it to the end! You can embark on your own MVPA 
% adventures now or check out one of the other tutorials.

%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
% The Kernel Ridge model is denoted as kernel_ridge. The kernel is one of
% the model's hyperparameters. 
cfg = [];
cfg.model   = 'kernel_ridge';
cfg.hyperparameter = [];
cfg.hyperparameter.kernel = 'polynomial';

perf = mv_regress(cfg, X, y);

close all
plot(time, perf)
xlabel('Time'), ylabel('MAE'), title('Kernel Ridge with polynomial kernel')

%% SOLUTION TO EXERCISE 2
% Mean Squared Error is denoted as either 'mse' or 'mean_squared_error'
cfg = [];
cfg.metric  = 'mse';
perf = mv_regress(cfg, X, y);

close all
plot(time, perf)

%% SOLUTION TO EXERCISE 3
% To define a holdout set, we need to set cv to 'holdout'. Then cfg.p
% defines the proportion of data in the test set. 20% corresponds to 0.2.
% In each repetition, we now only have a single fold, since we use a single
% holdout set as test data.
cfg = [];
cfg.cv      = 'holdout';
cfg.p       = 0.2;
perf = mv_regress(cfg, X, y);

%% SOLUTION TO EXERCISE 4
cfg = [];
cfg.metric  = {'mse' 'mae' 'r_squared'};
[~, result] = mv_regress(cfg, X, y);

mv_plot_result(result, time)

%% SOLUTION TO EXERCISE 5
% We only need to re-calculate the predictions y and then train/test the
% three regression models again
y = 2*mod(x, 3) + 0.4 * x; % SAWTOOTH FUNCTION
y_plus_noise  = y + randn(length(y), 1);

% train and test the regression models
model = train_ridge(param_ridge, x, y_plus_noise);
y_ridge = test_ridge(model, x);

model = train_kernel_ridge(param_krr, x, y_plus_noise);
y_kernel_ridge = test_kernel_ridge(model, x);

model = train_libsvm(param_svr, x, y_plus_noise);
y_svr = test_libsvm(model, x);

% Plot the data and the predictions of the three models in a plot
figure,hold on
plot(x,y, 'r-')  % true signal 
plot(x,y_plus_noise, 'ko')  % data (signal plus noise)
plot(x, y_ridge, 'b')   % ridge prediction
plot(x, y_kernel_ridge, 'k')   % kernel ridge prediction
plot(x, y_svr, 'g')   % SVR prediction

legend({'Sawtooth function (signal)' 'Data (signal+noise)' 'Ridge regression' 'Kernel ridge' 'SVR'})
title('Predictions for sawtooth function')
% Again, Ridge underfits the data with a line. Better results are
% obtained with Kernel Ridge and SVR, which smoothly approximate the
% sawtooth function.
