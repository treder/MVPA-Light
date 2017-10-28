%%% Classification across time using the mv_classify_across_time function
%%% In this function, we need data with a time dimension [samples x
%%% features x time points]. Then, cross-validation is run 

clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

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
cfg_LDA.metric     = 'acc';

% We are interested in comparing LDA and Logistic Regression (LR). To this 
% end, we setup a configuration struct for logreg as well. Again, the
% lambda parameter is optimised automatically.
cfg_LR =  cfg_LDA;
cfg_LR.classifier = 'logreg';
cfg_LR.param      = struct('lambda','auto' );

%% Run classification across time
auc_LDA = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
auc_LR = mv_classify_across_time(cfg_LR, dat.trial, clabel);

%% Plot classification accuracy across time
close all
mv_plot_1D([],dat.time, cat(2,auc_LDA,auc_LR) )
ylabel(cfg_LDA.metric)
legend({'LDA' 'LR'})

%% Classification across time for all subjects
nSbj = 3;
acc = cell(nSbj,1);         % classification accuracies for all subjects
auc = cell(nSbj,1);         % AUC values for all subjects

cfg_LDA.metric  = 'auc';

for nn=1:nSbj
    
    % Load dataset
    [dat,clabel] = load_example_data(['epoched' num2str(nn)]);

    % Run classification across time
    auc{nn} = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
    
end

auc = cat(2,auc{:});

% Average and standard error of classifier performance across subjects
av_auc = mean(auc,2);
se_auc = std(auc,[],2)/sqrt(nSbj);

%% Plot results
close all
subplot(1,2,1)
    mv_plot_1D([],dat.time,auc)
    title('Single-subject AUCs')
    legend(arrayfun(@(x) {['Subject ' num2str(x)]},1:nSbj))
subplot(1,2,2)
    mv_plot_1D([],dat.time, av_auc, se_auc)
    title('Grand average AUC')
    ylabel('AUC')

