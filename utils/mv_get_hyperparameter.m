function param = mv_get_hyperparameter(model, param)
% Returns a struct with default hyperparameters for a given 
% classifier or regression model. 
%
%Usage:
% param = mv_get_hyperparameter(model, <param>)
% 
%Parameters:
% model          - [string] the classifier or regression model (eg 'lda')
% param          - [struct] (optional) containing hyperparameters. The struct
%                  is filled up with default values for non-existing
%                  fields, but existing values are not overwritten. If
%                  param is not provided, all values are set to default
%
%Output:
% param          - [struct] with default hyperparameters

if nargin < 2 || ~isstruct(param)
    param = struct();
end

switch(model)

    %%% --- Defaults for classifiers ---
    case 'ensemble'
        mv_set_default(param,'learner','lda');
        mv_set_default(param,'learner_param',mv_get_hyperparameter(param.learner));
        mv_set_default(param,'nsamples', 0.5);
        mv_set_default(param,'nfeatures', 0.2);
        mv_set_default(param,'nlearners', 100);
        mv_set_default(param,'stratify', 1);
        mv_set_default(param,'bootstrap', 1);
        mv_set_default(param,'strategy', 'vote');
        mv_set_default(param,'simplify', false);
        
    case 'kernel_fda'
        mv_set_default(param,'reg','shrink');
        mv_set_default(param,'lambda',10e-5);
        mv_set_default(param,'kernel','linear');
%         mv_set_default(param,'regularize_kernel',10e-10);
        
        % parameters for specific kernels
        mv_set_default(param,'gamma','auto'); % RBF and polynomial kernel regularization parameter
        mv_set_default(param,'coef0',1);    % polynomial kernel
        mv_set_default(param,'degree',3);   % degree of polynomial kernel

    case 'lda'
        mv_set_default(param,'reg','shrink');
        mv_set_default(param,'lambda','auto');
        mv_set_default(param,'lambda_n',1e-12);
        mv_set_default(param,'form','auto');
        mv_set_default(param,'prob',0);
        mv_set_default(param,'scale',0);
        mv_set_default(param,'k',5);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'evtol',10^8);
    
    case 'libsvm'
        mv_set_default(param,'svm_type',0);
        mv_set_default(param,'kernel','rbf');
        mv_set_default(param,'degree',3);
        mv_set_default(param,'gamma',[]); % default is 1/numFeatures but since we don't know the features we set it to empty here [it's taken care of in LIBSVM then]
        mv_set_default(param,'coef0',0);
        mv_set_default(param,'cost',1);
        mv_set_default(param,'nu',0.5);
        mv_set_default(param,'epsilon',0.1);
        mv_set_default(param,'cachesize',100);
        mv_set_default(param,'eps',0.001);
        mv_set_default(param,'shrinking',1);
        mv_set_default(param,'probability_estimates',0);
        mv_set_default(param,'weight',1);
        mv_set_default(param,'cv',[]);
        mv_set_default(param,'quiet',1);
        
    case 'liblinear'
        mv_set_default(param,'type',0);
        mv_set_default(param,'cost',1);
        mv_set_default(param,'epsilon',0.1);
        mv_set_default(param,'eps',[]);  % use the defaults in LIBLINEAR
        mv_set_default(param,'bias',-1);
        mv_set_default(param,'weight',[]);
        mv_set_default(param,'cv',[]);
        mv_set_default(param,'c',[]);
        mv_set_default(param,'quiet',1);
        
    case 'logreg'
        mv_set_default(param,'reg','logf');
        mv_set_default(param,'bias',100);
        mv_set_default(param,'correct_bias', true);
        mv_set_default(param,'weights', []);
        mv_set_default(param,'lambda',1);
        mv_set_default(param,'max_iter',400);
        mv_set_default(param,'tolerance',1e-6);
        mv_set_default(param,'k',5);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'predict_regularization_path',0);
        mv_set_default(param,'polyorder',3);
        
    case 'multiclass_lda'
        mv_set_default(param,'reg','shrink');
        mv_set_default(param,'lambda','auto');
         
    case 'naive_bayes'
        mv_set_default(param,'prior','equal');
        
    case 'svm'
        mv_set_default(param,'bias','auto');
        mv_set_default(param,'c',1);
        mv_set_default(param,'kernel','linear'); % 'poly' 'rbf'
        mv_set_default(param,'prob',0);
        mv_set_default(param,'regularize_kernel',10e-10);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'k',3);
        mv_set_default(param,'tolerance',0.01);
        mv_set_default(param,'shrinkage_multiplier',1);
        
        % parameters for specific kernels
        mv_set_default(param,'gamma','auto'); % RBF and polynomial kernel
        mv_set_default(param,'coef0',1);    % polynomial kernel
        mv_set_default(param,'degree',2);   % degree of polynomial kernel

        %%% --- Defaults for regression models ---
    case 'ridge'
        mv_set_default(param,'lambda',1);
        mv_set_default(param,'form','auto');
        mv_set_default(param,'k',5);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'evtol',10^8);
        mv_set_default(param,'correlation_bound', []);

    case 'kernel_ridge'
        mv_set_default(param,'lambda',1);
        mv_set_default(param,'k',5);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'kernel','rbf');
        % parameters for specific kernels
        mv_set_default(param,'gamma','auto'); % RBF and polynomial kernel
        mv_set_default(param,'coef0',1);    % polynomial kernel
        mv_set_default(param,'degree',3);   % degree of polynomial kernel
        mv_set_default(param,'correlation_bound', []);

    case 'svr'  
        % note: uses the same libary as 'libsvm' just default parameters
        % for svm_type is different 
        mv_set_default(param,'svm_type',3);
        mv_set_default(param,'kernel','rbf');
        mv_set_default(param,'degree',3);
        mv_set_default(param,'gamma',[]); % default is 1/numFeatures but since we don't know the features we set it to empty here [it's taken care of in LIBSVM then]
        mv_set_default(param,'coef0',0);
        mv_set_default(param,'cost',1);
        mv_set_default(param,'nu',0.5);
        mv_set_default(param,'epsilon',0.1);
        mv_set_default(param,'cachesize',100);
        mv_set_default(param,'eps',0.001);
        mv_set_default(param,'shrinking',1);
        mv_set_default(param,'probability_estimates',0);
        mv_set_default(param,'weight',1);
        mv_set_default(param,'cv',[]);
        mv_set_default(param,'quiet',1);
    otherwise, error('Unknown model ''%s''',model)
end
