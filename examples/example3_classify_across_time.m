%%% Classification across time using the mv_classify_across_time function
clear all

% Load data (in /examples folder)
load('epoched3')
dat.trial = double(dat.trial);

% Create class labels (1's and 2's)
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;   % Class 1: attended deviants
clabel(~attended_deviant) = 2;   % Class 2: unattended deviants

% For Logistic regression, it is important that the data are scaled well.
% We therefore apply z-scoring.
dat.trial = zscore(dat.trial,[],1);

%% Calculate and plot ERP for attended and unattended deviants

% ERP for each condition
erp_attended = squeeze(mean(dat.trial(attended_deviant,:,:)));
erp_unattended = squeeze(mean(dat.trial(~attended_deviant,:,:)));

% Plot
plot(dat.time, erp_attended, 'r'), hold on
plot(dat.time, erp_unattended, 'g')
grid on

%% Setup configuration struct for LDA and Logistic Regression

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 10 repetitions. As classifier, we
% use LDA. The value of the regularisation parameter lambda is determined 
% automatically.
cfg_LDA =  [];
cfg_LDA.CV         = 'kfold';
cfg_LDA.K          = 5;
cfg_LDA.repeat     = 5;
cfg_LDA.classifier = 'lda';
cfg_LDA.param      = struct('lambda','auto');
cfg_LDA.verbose    = 1;
cfg_LDA.metric     = 'auc';

% We are interested in comparing LDA and Logistic Regression (LR). To this end,
% we setup a configuration struct for logreg as well.
cfg_LR =  cfg_LDA;
cfg_LR.classifier = 'logreg';
cfg_LR.param      = struct('lambda',logspace(-6,2,10));

cfg_SVM =  cfg_LDA;
cfg_SVM.classifier = 'svm';
cfg_SVM.param      = struct('lambda',logspace(-6,2,10));

%% Classification across time
acc_LDA = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
acc_LR = mv_classify_across_time(cfg_LR, dat.trial, clabel);
acc_SVM = mv_classify_across_time(cfg_SVM, dat.trial, clabel);

close all
mv_plot_1D([],dat.time,cat(2,acc_LDA,acc_LR,acc_SVM))
ylabel(cfg_LDA.metric)
legend({'LDA' 'LR' 'SVM'})

%% Classification across time for all subjects
nSbj = 3;
acc = cell(nSbj,1);         % classification accuracies for all subjects
auc = cell(nSbj,1);         % AUC values for all subjects

% As performance metrics, we calculate both classification accuracy and AUC
cfg_LDA.metric  = {'acc' 'auc'};

for nn=1:nSbj
    
    load(['epoched' num2str(nn)] )
    
    % Create class labels (1's and 2's)
    clabel = zeros(nTrial, 1);
    clabel(attended_deviant)  = 1;   % Class 1: attended deviants
    clabel(~attended_deviant) = 2;   % Class 2: unattended deviants
    
    % Run classification across time
    [acc{nn}, auc{nn}] = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
end

acc = cat(2,acc{:});
auc = cat(2,auc{:});

% Average and standard error of classifier performance across subjects
av_acc = mean(acc,2);
se_acc = std(acc,[],2)/sqrt(nSbj);
av_auc = mean(auc,2);
se_auc = std(auc,[],2)/sqrt(nSbj);

%% Plot results
close all
subplot(1,3,1)
    mv_plot_1D([],dat.time,acc)
    title('Single-subject accuracies')
    legend(arrayfun(@(x) {['Subject ' num2str(x)]},1:nSbj))
subplot(1,3,2)
    mv_plot_1D([],dat.time,av_acc, se_acc)
    title('Grand average classification accuracy')
subplot(1,3,3)
    mv_plot_1D([],dat.time,av_auc, se_auc)
    title('Grand average AUC')
    ylabel('AUC')

