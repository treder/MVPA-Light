%% TODO doesn't work correctly when there's two datasets X and X2 with labels and labels2
%% NEEDS FIXING

error('needs fixing')
function [acc] = mv_classify_timextime_two_datasets(cfg, X, labels, X2, labels2)
% Time x time generalisation. A classifier is trained on the first dataset 
% X with corresponding labels and validated on an independent dataset
% X2 with corresponding labels2.
%
% Usage:
% cf = mv_classify_timextime(cfg,X,labels)
% cf = mv_classify_timextime(cfg,X,labels,X2)
%
%Parameters:
% X              - [number of samples x number of features x number of time points]
%                  data matrix.
% labels         - [number of samples] vector of class labels containing
%                  1's (class 1) and -1's (class 2)
% X2             - second data matrix (optional). Should have the same number of
%                  samples and number of features as X, but can have 
%                  a different number of time points.
% labels2        - second label vector (optional)
%
% If two different datasets X and X2 are provided, the classifier is
% trained on X and tested on X2. No cross-validation is required.
% If only X is provided, both training and testing is performed on X.
% Cross-validation should then be used to obtain an out-of-sample estimate
% of the test performance.
%
% cfg          - struct with optional parameters:
% .classifier   - name of classifier, needs to have according train_ and test_
%                 functions (default 'lda')
% .param        - struct with parameters passed on to the classifier train
%                 function (default [])
% .time1        - indices of training time points (by default all time
%                 points in X are used)
% .time2        - indices of test time points (by default all time points
%                 in X or X2 are used)
% .balance      - for imbalanced data with a minority and a majority class.
%                 'oversample' oversamples the minority class
%                 'undersample' undersamples the minority class
%                 such that both classes have the same number of samples
%                 (default 'none'). Note that for we undersample at the
%                 level of the repeats, whereas we oversample within each
%                 training set (for an explanation see mv_balance_classes)
% .replace      - if balance is set to 'oversample' or 'undersample',
%                 replace deteremines whether data is drawn with
%                 replacement (default 1)
% .normalise    - for evoked data is it recommended to normalise the samples
%                 across trials, for each time point and each sensor
%                 separately, using 'zscore' or 'demean' (default 'none')
% .verbose      - print information on the console (default 1)
%
% Returns:
% acc           - time1 x time2 classification accuracy matrix

% (c) Matthias Treder 2017

if nargin < 4 || isempty(X2)
    X2 = X;
    labels2 = labels;
end

[~,~,nTime1] = size(X);
nTime2 = size(X2,3);

mv_setDefault(cfg,'classifier','lda');
mv_setDefault(cfg,'param',[]);
mv_setDefault(cfg,'time1',1:nTime1);
mv_setDefault(cfg,'time2',1:nTime2);
mv_setDefault(cfg,'normalise','none');
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

nTime1 = numel(cfg.time1);
nTime2 = numel(cfg.time2);

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

%% Normalise 
if strcmp(cfg.normalise,'zscore')
    X = zscore(X,[],1);
    X2 = zscore(X,[],1);
elseif strcmp(cfg.normalise,'demean')
    X  = X  - repmat(mean(X,1), [nSam 1 1]);
    X2 = X2 - repmat(mean(X,1), [nSam 1 1]);
end

%% Time x time generalisation

% Output matrix with average classification accuracy
acc= zeros(nTime1,nTime2);

% Save original data and labels in case we do over/undersampling
X_orig = X;
labels_orig = labels;

    % No cross-validation, just train and test once for each
    % training/testing time
    
    for t1=1:nTime1          % ---- Training time ----
        % Training data
        Xtrain= squeeze(X(:,:,cfg.time1(t1)));
        
        % Train classifier
        cf= train_fun(Xtrain, labels, cfg.param);
        
        for t2=1:nTime2      % ---- Testing time ----

            % Test data
            Xtest=  squeeze(X2(:,:,cfg.time2(t2)));
            
            % Obtain the predicted class labels
            predlabel = test_fun(cf,Xtest);
            
            % Sum number of correctly predicted labels
            acc(t1,t2)= acc(t1,t2) + sum(predlabel(:) == labels(:));    
        end
    end
    
    acc = acc / nSam;

