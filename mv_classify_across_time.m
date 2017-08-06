function [perf,cfs] = mv_classify_across_time(cfg, X, labels)
% Classification across time. A classifier is trained and validate for
% different time points in the dataset X. Cross-validation should be used
% to get a realistic estimate of classification performance.
%
% Usage:
% [acc,cfs] = mv_classify_across_time(cfg,X,labels)
%
%Parameters:
% X              - [number of samples x number of features x number of time points]
%                  data matrix.
% labels         - [number of samples] vector of class labels containing
%                  1's (class 1) and -1's (class 2)
%
% cfg          - struct with hyperparameters:
% .classifier   - name of classifier, needs to have according train_ and test_
%                 functions (default 'lda')
% .param        - struct with parameters passed on to the classifier train
%                 function (default [])
% .metric       - classifier performance metric, default 'acc'. See
%                 mv_calculate_metric. Multiple metrics can be requested by
%                 providing a cell array e.g. {'acc' 'dval'}
% .CV           - perform cross-validation, can be set to
%                 'kfold' (recommended) or 'leaveout' (not recommended
%                 since it has a higher variance than k-fold) (default
%                 'none')
% .K            - number of folds (the K in K-fold cross-validation). 
%                 For leave-one-out, K should be 1. (default 10 for kfold,
%                 1 for leave-one-out)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds. Only useful for CV = 'kfold'
%                 (default 1)
% .time         - indices of training time points (by default all time
%                 points in X are used)
% .balance      - for imbalanced data with a minority and a majority class.
%                 'oversample' oversamples the minority class
%                 'undersample' undersamples the minority class
%                 such that both classes have the same number of samples
%                 (default 'none'). Note that for we undersample at the
%                 level of the repeats, whereas we oversample within each
%                 training set (for an explanation see mv_balance_classes).
%                 You can also give an integer number for undersampling.
%                 The samples will be reduced to this number. Note that
%                 concurrent over/undersampling (oversampling of the
%                 smaller class, undersampling of the larger class) is not
%                 supported at the moment
% .replace      - if balance is set to 'oversample' or 'undersample',
%                 replace deteremines whether data is drawn with
%                 replacement (default 1)
% .verbose      - print information on the console (default 1)
%
% Returns:
% perf          - [time x 1] vector of classifier performances. If multiple
%                 metrics were requested, a cell array with vectors of
%                 classifier performance metrics is provided instead
% cfs           - [repeat x K x time] cell array containing the classifiers
%                 trained in each repetition and each fold for each time
%                 point
%
% Note: For time x time generalisation, use mv_classify_timextime

% (c) Matthias Treder 2017

mv_setDefault(cfg,'classifier','lda');
mv_setDefault(cfg,'param',[]);
mv_setDefault(cfg,'metric','acc');
mv_setDefault(cfg,'CV','none');
mv_setDefault(cfg,'repeat',5);
mv_setDefault(cfg,'time',1:size(X,3));
mv_setDefault(cfg,'verbose',0);

% Balance the data using oversampling or undersampling
mv_setDefault(cfg,'balance','none');
mv_setDefault(cfg,'replace',1);

if strcmp(cfg.CV,'kfold')
    mv_setDefault(cfg,'K',10);
else
    mv_setDefault(cfg,'K',1);
end

[~,~,labels] = mv_check_labels(labels);

nTime = numel(cfg.time);

if nargout>1
    cfs= cell(cfg.repeat,cfg.K,nTime);
end

% Number of samples in the classes
N1 = sum(labels == 1);
N2 = sum(labels == -1);

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% Prepare performance metrics
if ~iscell(cfg.metric)
    cfg.metric = {cfg.metric};
end

nMetrics = numel(cfg.metric);
perf= repmat( {zeros(nTime,1)}, [1 nMetrics]);
    
%% Classify across time

% Save original data and labels in case we do over/undersampling
X_orig = X;
labels_orig = labels;

if ~strcmp(cfg.CV,'none')
    if cfg.verbose, fprintf('Using %s cross-validation (K=%d) with %d repetitions.\n',cfg.CV,cfg.K,cfg.repeat), end
    
    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.verbose, fprintf('\nRepetition #%d. Fold ',rr), end
        
        % Undersample data if requested. We undersample the classes within the
        % loop since it involves chance (samples are randomly over-/under-
        % sampled) so randomly repeating the process reduces the variance
        % of the result
        if strcmp(cfg.balance,'undersample')
            [X,labels] = mv_balance_classes(X_orig,labels_orig,cfg.balance,cfg.replace);
        elseif isnumeric(cfg.balance)
            if ~all( cfg.balance <= [N1,N2])
                error(['cfg.balance is larger [%d] than the samples in one of the classes [%d, %d]. ' ...
                    'Concurrent over- and undersampling is currently not supported.'],cfg.balance,N1,N2)
            end
            % Sometimes we want to undersample to a specific
            % number (e.g. to match the number of samples across
            % subconditions)
            [X,labels] = mv_balance_classes(X_orig,labels_orig,cfg.balance,cfg.replace);
        end
        
        CV= cvpartition(labels,cfg.CV,cfg.K);
        
        for ff=1:cfg.K                      % ---- CV folds ----
            if cfg.verbose, fprintf('%d ',ff), end
            
            % Train data
            Xtrain = X(CV.training(ff),:,:,:);
            
            % Split labels into training and test
            trainlabels= labels(CV.training(ff));
            testlabels= labels(CV.test(ff));

            % Oversample data if requested. It is important to oversample 
            % only the *training* set to prevent overfitting (see
            % mv_balance_classes for an explanation)
            if strcmp(cfg.balance,'oversample')
                [Xtrain,trainlabels] = mv_balance_classes(Xtrain,trainlabels,cfg.balance,cfg.replace);
            end
            
            for tt=1:nTime           % ---- Train and test time ----
                % Train and test data for time point tt
                Xtrain_tt= squeeze(Xtrain(:,:,cfg.time(tt)));
                Xtest= squeeze(X(CV.test(ff),:,cfg.time(tt)));
                
                % Train classifier
                cf= train_fun(Xtrain_tt, trainlabels, cfg.param);
                
                % Obtain the performance metrics
                for mm=1:nMetrics
                    perf{mm}(tt) = perf{mm}(tt) + mv_calculate_metric(cfg.metric{mm}, cf, test_fun, Xtest, testlabels, 1);
                end
                
                if nargout>1
                    cfs{rr,ff,tt} = cf;
                end
            end
        end
    end
       
    % We have to divide the classifier performance by the number of
    % repetitions x number of folds to get the correct mean performance 
    for mm=1:nMetrics
        perf{mm} = perf{mm} / (cfg.repeat * cfg.K);
    end
    
else
    
    % No cross-validation, just train and test once for each
    % training/testing time. This gives the classification performance for
    % the training set, but it may lead to overfitting and thus to an
    % artifically inflated performance.
    
    % Rebalance data using under-/over-sampling if requested
    if ~strcmp(cfg.balance,'none')
        [X,labels] = mv_balance_classes(X_orig,labels_orig,cfg.balance,cfg.replace);
    end
    
    for tt=1:nTime          % ---- Train and test time ----
        % Train and test data
        Xtraintest= squeeze(X(:,:,cfg.time(tt)));
        
        % Train classifier
        cf= train_fun(Xtraintest, labels, cfg.param);
        
        % Obtain the performance metrics
        for mm=1:nMetrics
            perf{mm}(tt) = mv_calculate_metric(cfg.metric{mm}, cf, test_fun, Xtraintest, labels, 1);
        end
    end
    
    if nargout>1
        cfs= cf;
    end
end

% Un-nest the cell array if only one performance metric was requested
if nMetrics==1
    perf = perf{1};
end
