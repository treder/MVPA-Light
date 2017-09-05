%%% Train and test logistic regression.

close all
clear all

% Load data (in /examples folder)
load('epoched2')

% Create class labels (1's and 2's)
label = zeros(nTrial, 1);
label(attended_deviant)  = 1;   % Class 1: attended deviants
label(~attended_deviant) = 2;  % Class 2: unattended deviants

% Average activity in 0.6-0.8 interval (see example1)
ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

%% Logistic regression

param = mv_classifier_defaults('logreg');

cf = train_logreg(X, truelabel, param);

%% Hyperparameter Alpha: classification accuracy and computation duration
%%% Investigate the effects on classification performance and computation
%%% duration of the hyperparameter alpha. The parameter is varied from
%%% 10^-10 to 1.
%%% For larger alpha terms, the computation takes quite a while!

% alphas = 10.^[-10:-3];
alphas = [10.^[-10:-1], 0.2, 0.5, 1];

acc = zeros(numel(alphas),1);
time = zeros(numel(alphas),1);

% Get default parameters for logistic regression
param = mv_classifier_defaults('logreg');

% Setup cfg for cross-validation
cfg = [];
cfg.classifier  = 'logreg';
cfg.K           = 5;
cfg.repeat      = 2;
cfg.balance     = 'undersample';

for aa=1:numel(alphas)
    fprintf('Lambda=%2.6f\n',alphas(aa))
    
    % Vary the lambda parameter
    param.alpha = alphas(aa);
    cfg.param = param;
    
    tic
    acc(aa) = mv_crossvalidate(cfg, X, label);
    time(aa) = toc;
end

figure
subplot(1,2,1)
%     semilogx(alphas, acc)
    plot(alphas, acc)
    xlabel('Alpha'),ylabel('Classification accuracy')
subplot(1,2,2)
    semilogx(alphas, time)
    xlabel('Alpha'),ylabel('Computation time [s]')
    
    
%% Hyperparameter lambda: Deviance and sparsity
%%% If alpha is set to an intermediate or large value e.g. alpha > 0.2, the
%%% coefficient vector tends to become more sparse. The amount of sparsity
%%% (=number of zero coefficients) is determined by the amount of
%%% regularisation lambda.

param = mv_classifier_defaults('logreg');
param.alpha = 10^-8;%0.5;

[cf, b, stats] = train_logreg(X,label,param);

figure
subplot(1,2,1),
    lassoPlot(b,stats,'plottype','CV');
    title('Deviance as a function of lambda')
subplot(1,2,2),
    semilogx(stats.lambda, sum(b))
    title('Sparsity as a function of lambda')
    xlabel('Lambda'),ylabel('Nr. of zero coefficients')