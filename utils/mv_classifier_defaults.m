function param = mv_classifier_defaults(classifier,param)
% Sets the default hyper-parameters for each classifier. 
%
% Usage:
% param = mv_classifier_default_param(classifier)
% param = mv_classifier_default_param(classifier, param)
% 
%Parameters:
% classifier     - string specifying the classifier (e.g. 'lda')
% param          - optional struct containing hyperparameters. The struct
%                  is filled up with default values for non-existing
%                  fields, but existing values are not overwritten. If
%                  param is not provided, all values are set to default
%
%Output:
% param  - struct with default values added

if nargin < 2 || ~isstruct(param)
    param = struct();
end

switch(classifier)
    
    case 'lda'
        mv_setDefault(param,'lambda','auto');
        mv_setDefault(param,'prob',0);
        mv_setDefault(param,'scale',0);
        
    case 'logist'
        mv_setDefault(param,'v','mean');
        mv_setDefault(param,'lambda',[]);
        mv_setDefault(param,'eigvalratio',[]);

    case 'logreg'
        mv_setDefault(param,'normalise',0);
        mv_setDefault(param,'intercept',1);
        % mv_setDefault(param,'lambda',2.^[-10:10]);
        mv_setDefault(param,'lambda',1);
        mv_setDefault(param,'optim',optimoptions('fsolve','Algorithm',...
            'trust-region-dogleg',...
            'SpecifyObjectiveGradient',true,'Display','none') );

    case 'logreg_matlab'
        mv_setDefault(param,'alpha',0.01);
        mv_setDefault(param,'numLambda',100);
        mv_setDefault(param,'K',5);
        mv_setDefault(param,'nameval',{});

    case 'ensemble'
        mv_setDefault(param,'learner','lda');
        mv_setDefault(param,'learner_param',[]);
        mv_setDefault(param,'nSamples', 0.5);
        mv_setDefault(param,'nFeatures', 0.2);
        mv_setDefault(param,'nLearners', 500);
        mv_setDefault(param,'stratify', false);
        mv_setDefault(param,'replace', 1);
        mv_setDefault(param,'strategy', 'dval');
        mv_setDefault(param,'simplify', false);
        
    otherwise, error('Unknown classifier ''%s''',classifier)
end