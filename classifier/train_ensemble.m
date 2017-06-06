function cf = train_ensemble(X,labels,param)
% Trains an ensemble of classifiers. Uses many classifiers (aka weak
% learners) trained on subsets of the samples and subsets of the features
% (random subspaces).
%
% Usage:
% cf = train_ensemble(cfg,X,labels)
% 
%Parameters:
% X              - [number of samples x number of features] matrix of
%                  training samples
% labels         - [number of samples] vector of class labels containing 
%                  1's (class 1) and -1's (class 2)
%
% cfg: struct with parameters:
% .nSamples         - number of randomly subselected samples for each
%                     learner. Can be an integer number or a
%                     fraction, e.g. 0.1 means that 10% of the training 
%                     data is used for each learner. (Default 0.5)
% .nFeatures        - number of randomly subselected features. Can be an 
%                     integer number or a fraction, e.g. 0.1 means that 
%                     10% of the features are used for each learner. (Default 0.1)
% .nLearners        - number of learners (default 500)
% .strategy         - strategy for making decisions. If 'vote', the class
%                     label from each learner is obtained, then the class
%                     associated with the majority vote it taken (randomly
%                     choose a class in case of a draw). Many classifiers
%                     also provided decision values; in this case, the
%                     decision values can be averaged and the decision is
%                     taken according to the mean decision value, use
%                     'dval' in this case (default 'dval')
% .stratify         - if 1, takes care that the class proportions are
%                     preserved during the subselection of samples. Can
%                     also be given as a fraction specifying the fraction
%                     of class 1, e.g. 0.6 means that 60% of the samples
%                     are coming from class 1
% .replace          - if 1, samples are selected with replacement
%                     (this is also called bootstrapping), otherwise they
%                     are drawn without replacement (default 1)
% .learner          - type of learning algorithm, e.g. 'lda','logreg' etc. 
% .param            - struct with further parameters passed on to the learning
%                     algorithm (e.g. .param.gamma specifies the
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

% (c) Matthias Treder 2017

[N,F] = size(X);

% default settings
mv_setDefault(param,'learner','lda');
mv_setDefault(param,'param',[]);
mv_setDefault(param,'nSamples', 0.5);
mv_setDefault(param,'nFeatures', 0.2);
mv_setDefault(param,'nLearners', 500);
mv_setDefault(param,'stratify', false);
mv_setDefault(param,'replace', 1);
mv_setDefault(param,'strategy', 'dval');
mv_setDefault(param,'simplify', false);

% if fractions are given for nFeatures and nLearners, turn them into
% absolute numbers
if param.nSamples < 1
    param.nSamples= round(param.nSamples*N);
end
if param.nFeatures < 1
    param.nFeatures= round(param.nFeatures*F);
end

% if we want stratification, we need to calculate how many samples of each
% class we need in the subselected data
if param.stratify > 0
    % indices for class 1 and 2
    idx1= find(labels==1);
    idx2= find(labels==-1);
    % class proportions
    N1= numel(idx1);
    N2= numel(idx2);
    if param.stratify < 1 
        % fraction of desired samples from class 1 is provided
        p1= param.stratify;
    else
        % calculate fraction of desired samples from class 1 from data
        p1= N1/N;
    end
    % number of subselected samples from class 1 and 2
    Nsub1= round(param.nSamples * p1);
    Nsub2= param.nSamples - Nsub1;
end
    
%% Select random features for the learners
randomFeatures = sparse(false(F,param.nLearners));
for ll=1:param.nLearners
    randomFeatures(randperm(F,param.nFeatures),ll)=true;
end

%% Select random samples for the learners
% We have to consider different cases 
randomSamples = zeros(param.nSamples,param.nLearners);
if param.stratify
     if param.replace
        for ll=1:param.nLearners
            randomSamples(1:Nsub1,ll)=idx1(randi(N1,1,Nsub1));
        end
        for ll=1:param.nLearners
            randomSamples(Nsub1+1:end,ll)=idx2(randi(N2,1,Nsub2));
        end
    else % draw without replacement
        for ll=1:param.nLearners
            randomSamples(:,ll)=randperm(N,param.nSamples);
        end
    end
% no stratification, draw samples without caring for class labels
else 
    if param.replace
        for ll=1:param.nLearners
            randomSamples(:,ll)=randi(N,1,param.nSamples);
        end
    else % draw without replacement
        for ll=1:param.nLearners
            randomSamples(:,ll)=randperm(N,param.nSamples);
        end
    end
end

randomSamples = sort(randomSamples);

%% Train learner ensemble
cf = struct('randomFeatures',randomFeatures,'strategy',param.strategy,...
    'nLearners',param.nLearners,'simplify',param.simplify);
cf.train= eval(['@train_' param.classifier ]);
cf.test= eval(['@test_' param.classifier ]);

if param.simplify
    % In linear classifiers, the operation of the ensemble is equivalent to
    % the operation of a single classifier with appropriate weight w and
    % threshold b.
    % To obtain a single w, we pad all w's with zeros (for the discarded
    % features) and then add up the w's.
    cf.w = zeros(F,1);
    cf.b = 0;
    for ll=1:param.nLearners
        tmp = cf.train(param.param, X(randomSamples(:,ll),randomFeatures(:,ll)),labels(randomSamples(:,ll)));
        cf.w(randomFeatures(:,ll)) = cf.w(randomFeatures(:,ll)) + tmp.w;
        cf.b = cf.b + tmp.b;
    end
    cf.w = cf.w / param.nLeaners;
    cf.b = cf.b / param.nLeaners;
else
    % Initialise struct array of learners
    cf.classifier(param.nLearners) = cf.train(param.param, X(randomSamples(:,param.nLearners),randomFeatures(:,param.nLearners)),labels(randomSamples(:,param.nLearners)));
    
    % Get all the other learners
    for ll=1:param.nLearners-1
        cf.classifier(ll) = cf.train(param.param, X(randomSamples(:,ll),randomFeatures(:,ll)),labels(randomSamples(:,ll)));
    end
end







