%%% Classification across time using the mv_classify_across_time function.
%%% In this function, we need data with a time dimension [samples x
%%% features x time points]. Then, cross-validation is run for every time
%%% point separately, hence we obtain a classification performance score
%%% for every time point.

clear all

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched2');

%% Setup configuration struct for LDA and Logistic Regression

% Configuration struct for time classification with cross-validation. We
% perform 5-fold cross-validation with 2 repetitions. As classifier, we
% use LDA with its default settings.
cfg_LDA =  [];
cfg_LDA.cv              = 'kfold';
cfg_LDA.k               = 5;
cfg_LDA.repeat          = 2;
cfg_LDA.classifier      = 'lda';
cfg_LDA.metric          = 'accuracy';

% We are interested in comparing LDA and Logistic Regression (LR). To this 
% end, we setup a configuration struct for logreg as well. Again, we do not
% set the cfg.hyperparameter field so the default hyperparameters are used.
cfg_LR =  cfg_LDA;
cfg_LR.classifier       = 'logreg';

%% Run classification across time
[acc_LDA, result_LDA] = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
[acc_LR, result_LR] = mv_classify_across_time(cfg_LR, dat.trial, clabel);

%% Plot classification accuracy across time
close all
mv_plot_result({result_LDA, result_LR}, dat.time) % second argument is optional

%% Classification across time for all 3 subjects
nSbj = 3;
acc = cell(nSbj,1);         % classification accuracies for all subjects
auc = cell(nSbj,1);         % AUC values for all subjects
result = cell(nSbj,1);
cfg_LDA.metric  = {'precision' 'recall'};

for nn=1:nSbj
    
    % Load dataset
    [dat,clabel] = load_example_data(['epoched' num2str(nn)]);

    % Run classification across time
    [auc{nn}, result{nn}] = mv_classify_across_time(cfg_LDA, dat.trial, clabel);
    
    % Name the result (this will appear in the legend of the plot)
    result{nn}.name = sprintf('Subject #%d', nn);
    
end

% Plot 3 subjects together
close all
h = mv_plot_result(result, dat.time);

%% Plot average of three subjects 
% (shaded area is the standard deviation across subjects)
h = mv_plot_result(result, dat.time, 'combine','average');
