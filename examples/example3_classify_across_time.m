%%% Classification across time using the mv_classify_across_time function
%%% In this function, we need data with a time dimension [samples x
%%% features x time points]. Then, cross-validation is run 

clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

%% Setup configuration struct for LDA and Logistic Regression

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 2 repetitions. As classifier, we
% use LDA. The value of the regularisation parameter lambda is determined 
% automatically.
cfg_LDA =  [];
cfg_LDA.cv              = 'kfold';
cfg_LDA.k               = 5;
cfg_LDA.repeat          = 2;
cfg_LDA.classifier      = 'lda';
cfg_LDA.param           = [];       % sub-struct with hyperparameters for classifier
cfg_LDA.param.lambda    = 'auto';
cfg_LDA.metric          = 'accuracy';

% We are interested in comparing LDA and Logistic Regression (LR). To this 
% end, we setup a configuration struct for logreg as well. Again, the
% lambda parameter is optimised automatically.
cfg_LR =  cfg_LDA;
cfg_LR.classifier       = 'logreg';
cfg_LR.param            = [];       % sub-struct with hyperparameters for classifier
cfg_LR.param.lambda     = 'auto';

%% Run classification across time
[acc_LDA, result_LDA] = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
[acc_LR, result_LR] = mv_classify_across_time(cfg_LR, dat.trial, clabel);

%% Plot classification accuracy across time
close all
mv_plot_result({result_LDA, result_LR}, dat.time) % second argument is optional

%% Classification across time for all subjects
nSbj = 3;
acc = cell(nSbj,1);         % classification accuracies for all subjects
auc = cell(nSbj,1);         % AUC values for all subjects
result = cell(nSbj,1);
cfg_LDA.metric  = 'auc';

for nn=1:nSbj
    
    % Load dataset
    [dat,clabel] = load_example_data(['epoched' num2str(nn)]);

    % Run classification across time
    [auc{nn}, result{nn}] = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
    
end

% Plot 3 subjects and mean across subjects
close all
h = mv_plot_result(result, dat.time);
