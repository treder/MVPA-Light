%%% In this example we look at regression tasks. We will use the same
%%% datasets used in the classification examples. Since no response
%%% variable (e.g. reaction time) was recorded we will use the number of
%%% the trial (as a proxy for time in the experiment) as response variable.
%%% This can tell us whether there are changes in amplitude across the
%%% duration of the experiment. 

close all
clear all

% Load data (in /examples folder)
[dat,clabel,chans] = load_example_data('epoched1');

% keep only class 2
dat.trial = dat.trial(clabel==2, :, :);

% the response variable is simply the number of the trial
y = [1:size(dat.trial, 1)]';

% [dat2,clabel2] = load_example_data('epoched2');
% [dat3,clabel3] = load_example_data('epoched3');

% Plot data
close
h1= plot(dat.time, squeeze(mean(dat.trial, 1)), 'r'); hold on
grid on
xlabel('Time [s]'),ylabel('EEG amplitude')
title('ERP')

%% Regression on the [0.2 - 0.3] s ERP component

% mark the start and end of component in the plot
yl = ylim;
plot([0.2, 0.2], yl, 'k--'); 
plot([0.3, 0.3], yl, 'k--');

time_points = find( (dat.time >= 0.2) & (dat.time <= 0.3) );

X = squeeze(mean(dat.trial(:, :, time_points), 3));

% Set up the structure with options for mv_regress
cfg = [];
cfg.model   = 'ridge';               % ridge regression (inludes linear regression)
cfg.hyperparameter.lambda = [ 0, 1, 2, 10];
cfg.metric  = {'mse', 'mean_absolute_error'}; % can be abbreviated as 'mae'
cfg.dimension_names = {'samples' 'channels'};

% Call mv_regress to perform the regression 
[perf, result] = mv_regress(cfg, X, y);

mv_plot_result(result)

%% Regression across time
% Perform the same regression but this time for every time point, yielding
% MAE as a function of time

% Set up the structure with options for mv_regress
cfg = [];
cfg.model   = 'ridge';
cfg.metric  = 'mae';                 % = mean absolute error
cfg.dimension_names = {'samples' 'channels', 'time points'};

[perf, result] = mv_regress(cfg, dat.trial, y);

% ax = mv_plot_1D(dat.time, perf, result.perf_std, 'ylabel', cfg.metric)
mv_plot_result(result, dat.time)


%% Compare ridge regression / kernel ridge / Support Vector Regression
% To illustrate how kernels tackle non-linear problems, we will
% here create an 1-dimensional non-linear dataset. We will then train ridge
% regression, kernel ridge, and Support Vector Regression (SVR) models and
% compare them. 
% Note: The SVR model requires an installation of LIBSVM, see
% train_libsvm.m for details

x = linspace(0, 12, 100)';
y = -.1*x.^2 + 3*sin(x);     % SINUSOID WITH QUADRATIC TREND
% y = 2*mod(x, 3) + 0.4 * x; % SAWTOOTH FUNCTION
y_plus_noise  = y + randn(length(y), 1);

close all
plot(x,y, 'r', 'LineWidth', 2)
hold on
plot(x,y_plus_noise, 'ko')
legend({'Signal' 'Signal plus noise'})
title('True signal and data')

% Train ridge model and get predicted values 
param = mv_get_hyperparameter('ridge');
model = train_ridge(param, x, y);
y_ridge = test_ridge(model, x);

% Train kernel ridge model and get predicted values 
param = mv_get_hyperparameter('kernel_ridge');
% param.kernel = 'polynomial';
model = train_kernel_ridge(param, x, y);
y_kernel_ridge = test_kernel_ridge(model, x);

% Train SVR model and get predicted values.
% We will use the LIBSVM toolbox here, which supports both 
% classification (SVM) and regression (SVR).
param = mv_get_hyperparameter('libsvm');

% Set svm_type to 3 for support vector regression
param.svm_type = 3; 
model = train_libsvm(param, x, y);
y_svr = test_libsvm(model, x);

figure,hold on
% plot(x,y, 'r', 'LineWidth', 2)  % true signal
plot(x,y_plus_noise, 'ko')
plot(x, y_ridge, 'b')   % ridge prediction
plot(x, y_kernel_ridge, 'k')   % kernel ridge prediction
plot(x, y_svr, 'g')   % SVR prediction

legend({'Data' 'Ridge regression' 'Kernel ridge' 'SVR'})
title('Predictions')




