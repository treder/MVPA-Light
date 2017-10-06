% Play around with the logreg implementation
close all
clear all

% Load data (in /examples folder)
load('epoched3')
dat.trial = double(dat.trial);

% attenden_deviant contains the information about the trials. Use this to
% create the true class labels, indicating whether the trial corresponds to
% an attended deviant (1) or an unattended deviant (2).
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;  % Class 1: attended deviants
clabel(~attended_deviant) = 2;  % Class 2: unattended deviants

ival_idx = find(dat.time >= 0.6 & dat.time <= 0.8);

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival_idx),3));

% Get default hyperparameters
param = mv_classifier_defaults('lda');

% Train an LDA classifier
tic
cf = train_lda(param, X, clabel);
toc

%% -- Logistic regression
param = mv_classifier_defaults('logreg');
param.lambda = logspace(-6,3,10); % 2
param.plot = 0;
param.tolerance = 1e-6;
param.polyorder = 0;

tic
% profile on
cf = train_logreg(param, X, clabel);
% cf = train_logreg(param, X(:,[1,21]), clabel);
toc
% profile viewer
% profile off

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
