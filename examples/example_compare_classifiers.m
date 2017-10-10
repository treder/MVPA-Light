%%% Compare different classifiers for classification across time
clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched1');

%% Setup configuration struct for LDA and Logistic Regression

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 10 repetitions. As classifier, we
% use LDA. The value of the regularisation parameter lambda is determined 
% automatically.
cfg_LDA =  [];
cfg_LDA.CV         = 'kfold';
cfg_LDA.K          = 5;
cfg_LDA.repeat     = 2;
cfg_LDA.classifier = 'lda';
cfg_LDA.param      = struct('lambda','auto');
cfg_LDA.verbose    = 1;
cfg_LDA.metric     = 'auc';

% We are interested in comparing LDA and Logistic Regression (LR). To this end,
% we setup a configuration struct for logreg as well.
cfg_LR =  cfg_LDA;
cfg_LR.classifier = 'logreg';
% cfg_LR.param      = struct('lambda',logspace(-6,2,20));
cfg_LR.param      = struct('lambda',0.01 );

cfg_SVM =  cfg_LDA;
cfg_SVM.classifier = 'svm';
% cfg_SVM.param      = struct('lambda',logspace(-6,2,20));
cfg_SVM.param      = struct('lambda',1);

cfg_LIBSVM =  cfg_LDA;
cfg_LIBSVM.classifier = 'libsvm';
cfg_LIBSVM.param = struct('kernel_type',0);

cfg_LIBLINEAR=  cfg_LDA;
cfg_LIBLINEAR.classifier = 'liblinear';
cfg_LIBLINEAR.param = struct('type',0,'bias',1); %,'C',1);

%% Classification across time
rng(1),tic,acc_LDA = mv_classify_across_time(cfg_LDA, dat.trial, clabel);toc
rng(1),tic,acc_LR = mv_classify_across_time(cfg_LR, dat.trial, clabel);toc
% rng(1),tic,acc_SVM = mv_classify_across_time(cfg_SVM, zscore(dat.trial), clabel);toc
rng(1),tic,acc_LIBSVM = mv_classify_across_time(cfg_LIBSVM, dat.trial, clabel);toc
rng(1),tic,acc_LIBLINEAR = mv_classify_across_time(cfg_LIBLINEAR, dat.trial, clabel);toc

% dat.trial(200,:) = 100 *dat.trial(200,:);
%%
close all
% mv_plot_1D([],dat.time,cat(2,acc_LDA,acc_LR,acc_SVM,acc_LIBSVM,acc_LIBLINEAR))
mv_plot_1D([],dat.time,cat(2,acc_LDA,acc_LR,acc_LIBSVM,acc_LIBLINEAR))
ylabel(cfg_LDA.metric)
% legend({'LDA' 'LR' 'SVM' 'LIBSVM' 'LIBLINEAR'})
legend({'LDA' 'LR' 'LIBSVM' 'LIBLINEAR'})


