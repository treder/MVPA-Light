function cf = train_ensemble(param,X,clabel)
% Trains an ensemble of classifiers. Uses many classifiers (aka weak
% learners) trained on subsets of the samples and subsets of the features
% (random subspaces).
%
% Usage:
% cf = train_ensemble(param,X,clabel)
% 
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% param: struct with parameters:
% .nsamples         - number of randomly subselected samples for each
%                     learner. Can be an integer number or a
%                     fraction, e.g. 0.1 means that 10% of the training 
%                     data is used for each learner. (Default 0.5)
% .nfeatures        - number of randomly subselected features. Can be an 
%                     integer number or a fraction, e.g. 0.1 means that 
%                     10% of the features are used for each learner. (Default 0.1)
% .nlearners        - number of learners (default 100)
% .strategy         - strategy for making decisions. If 'vote', the class
%                     label from each learner is obtained, then the class
%                     associated with the majority vote it taken (randomly
%                     choose a class in case of a draw). Many classifiers
%                     also provided decision values; in this case, the
%                     decision values can be averaged and the decision is
%                     taken according to the mean decision value, use
%                     'dval' in this case. Note that 'dval' only works for
%                     binary classifiers (default 'vote')
% .stratify         - if 1, takes care that the class proportions are
%                     preserved during the subselection of samples. If 0,
%                     samples are randomly chosen which can lead to some
%                     learners not 'seeing' a particular class (default 1)
% .bootstrap        - if 1, samples are selected with replacement
%                     (this is also called bootstrapping), otherwise they
%                     are drawn without replacement (default 1)
% .learner          - type of learning algorithm, e.g. 'lda','logreg' etc.
%                     Any classifier with train and test functions can
%                     serve as a learner.
% .learner_param    - struct with further parameters passed on to the learning
%                     algorithm (e.g. cfg.hyperparameter.learner_param.lambda specifies the
%                     regularisation hyperparameter for LDA)
% .simplify         - for linear classifiers, the operation of the ensemble
%                     is again equivalent to a single linear classifier.
%                     Hence, a projection w and a threshold b can be
%                     calculated which can increase efficiency for
%                     prediction. Currently, this works for lda and logreg
%
%Output:
% cf - struct specifying the ensemble classifier
%

% (c) Matthias Treder 2017-2018

[N,F] = size(X);
nclasses = max(clabel);

% dval only works for binary classification problems
if strcmp(param.strategy, 'dval') && max(clabel)>2
    error(['strategy=''dval'' only works for binary classification problems. ' ...
        'For multi-class problems, set strategy=''vote'''])
end

% if fractions are given for nfeatures and nlearners, turn them into
% absolute numbers
if param.nsamples < 1
    param.nsamples= round(param.nsamples*N);
end
if param.nfeatures < 1
    param.nfeatures= round(param.nfeatures*F);
end

% if we want stratification, we need to calculate how many samples of each
% class we need in the subselected data
if param.stratify > 0
    
    % Indices of the samples for each class
    cidx = arrayfun( @(c) find(clabel==c), 1:nclasses, 'Un', 0);
    
    % total number of samples in each class
    Ntotal = arrayfun( @(c) sum(clabel==c), 1:nclasses);
    
    % number of selected samples from each class
    Nsub= floor(param.nsamples * Ntotal / N);
    
    % due to flooring above, Nsub might not add up to nsamples. If samples
    % are missing, equally add samples to the classes until the discrepancy
    % is gone
    addN = param.nsamples - sum(Nsub);
    cc = 1;
    while addN > 0
        Nsub(cc) = Nsub(cc) + 1;
        cc = mod(cc, nclasses)+1;
        addN = addN - 1;
    end
end

%% Get learner hyperparameters
param.learner_param = mv_get_hyperparameter(param.learner, param.learner_param);

%% Select random features for the learners
random_features = sparse(false(F,param.nlearners));
for ll=1:param.nlearners
    random_features(randperm(F,param.nfeatures),ll)=true;
end

%% Select random samples for the learners
% We have to consider different cases 
random_samples = zeros(param.nsamples,param.nlearners);
if param.stratify
    
    % We need to fill up the random_samples vector with samples belonging
    % to each class. Here, it is identified which indices belong to which
    % class
    class_sample_idx = zeros(nclasses,2);
    for cc=1:nclasses
        if cc==1, class_sample_idx(cc,1) = 1;
        else
            class_sample_idx(cc,1) = class_sample_idx(cc-1,2)+1;
        end
        class_sample_idx(cc,2) = sum(Nsub(1:cc));
    end
    
    for cc=1:nclasses
        for ll=1:param.nlearners
            if param.bootstrap % draw with replacement
                random_samples(class_sample_idx(cc,1):class_sample_idx(cc,2),ll) = cidx{cc}(randi(Ntotal(cc),1,Nsub(cc)));
            else % draw without replacement
                random_samples(class_sample_idx(cc,1):class_sample_idx(cc,2),ll) = cidx{cc}(randperm(Ntotal(cc),Nsub(cc)));
            end
        end
    end
else 
    % no stratification, draw samples without caring for class labels
    if param.bootstrap
        for ll=1:param.nlearners
            random_samples(:,ll)=randi(N,1,param.nsamples);
        end
    else % draw without replacement
        for ll=1:param.nlearners
            random_samples(:,ll)=randperm(N,param.nsamples);
        end
    end
end

random_samples = sort(random_samples);

%% Train learner ensemble
cf = struct('random_features',random_features,'strategy',param.strategy,...
    'nlearners',param.nlearners,'simplify',param.simplify, 'nclasses', nclasses);
cf.train= eval(['@train_' param.learner ]);
cf.test= eval(['@test_' param.learner ]);

if param.simplify
    % In linear classifiers, the operation of the ensemble is equivalent to
    % the operation of a single classifier with appropriate weight w and
    % threshold b.
    % To obtain a single w, one can pad all w's with zeros (for the discarded
    % features) and then add up the w's.
    cf.w = zeros(F,1);
    cf.b = 0;
    for ll=1:param.nlearners
        tmp = cf.train(X(random_samples(:,ll),random_features(:,ll)),clabel(random_samples(:,ll)),param.learner_param);
        cf.w(random_features(:,ll)) = cf.w(random_features(:,ll)) + tmp.w;
        cf.b = cf.b + tmp.b;
    end
    cf.w = cf.w / param.nlearners;
    cf.b = cf.b / param.nlearners;
else
    % Initialise struct array of learners
    cf.classifier(param.nlearners) = cf.train(param.learner_param, X(random_samples(:,param.nlearners),random_features(:,param.nlearners)),clabel(random_samples(:,param.nlearners)));
    
    % Train all the other learners
    for ll=1:param.nlearners-1
        cf.classifier(ll) = cf.train(param.learner_param, X(random_samples(:,ll),random_features(:,ll)),clabel(random_samples(:,ll)));
    end
end







