% Play around with the logreg implementation
close all
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched3');

ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Get default hyperparameters
param = mv_get_classifier_param('lda');

% Train an LDA classifier
tic
cf = train_lda(param, X, clabel);
toc

%% PCA on data
X = detrend(X);

[U,E] = eig(cov(X));

Xp = X*U;

%% -- Logistic regression
param = mv_get_classifier_param('logreg');
param.lambda = 1; %logspace(-6,2,30); % 2
% param.plot = 1;
param.tolerance = 1e-6;
param.polyorder = 2;

param.predict_regularisation_path = 0;

tic
rng(1)
% profile on
% cf = train_logreg(param, zscore(X), clabel);
% cf = train_logreg(param, zscore(Xp), clabel);

%%% Compare logreg on original X and on PCs - should be identical

cf = train_logreg(param, X, clabel);
cf2 = train_logreg(param, Xp, clabel);
% cf = train_logreg(param, X(:,[1,21]), clabel);
toc
% profile viewer
% profile off

[cf.w , U*cf2.w]

%% Copy back into train_logreg and run to compare the optimisation to the 
%% standard Matlab solver FSOLVE

%%% FSOLVE - 5-fold CV

% K = 5;
% CV = cvpartition(N,'KFold',K);
% ws_fsolve = zeros(nFeat, numel(cfg.lambda));
% fun = @(w) lr_gradient_tanh(w);
%
% tic
% for ff=1:K
%     X = X0(CV.training(ff),:);
%     YX = Y(CV.training(ff),CV.training(ff))*X;
%
%     % Sum of samples needed for the gradient
%     sumyx = sum(YX)';
%
%     for ll=1:numel(cfg.lambda)
%         lambda = cfg.lambda(ll);
%         if ll==1
%             ws_fsolve(:,ll) = fsolve(@(w) lr_gradient(w), w0, cfg.optim);
%         else
%             ws_fsolve(:,ll) = fsolve(@(w) lr_gradient(w), ws_fsolve(:,ll-1), cfg.optim);
%         end
%
%     end
% end
% toc
