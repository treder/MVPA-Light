function param = mv_get_classifier_param(classifier,param)
% Returns a parameter struct with default (hyper)parameters for a given 
% classifier. 
%
%Usage:
% param = mv_get_classifier_param(classifier, <param>)
% 
%Parameters:
% classifier     - string specifying the classifier (e.g. 'lda')
% param          - [optional] struct containing hyperparameters. The struct
%                  is filled up with default values for non-existing
%                  fields, but existing values are not overwritten. If
%                  param is not provided, all values are set to default
%
%Output:
% param  - struct with default parameter values

if nargin < 2 || ~isstruct(param)
    param = struct();
end

switch(classifier)
    
    case 'lda'
        mv_set_default(param,'reg','shrink');
        mv_set_default(param,'lambda','auto');
        mv_set_default(param,'prob',0);
        mv_set_default(param,'scale',0);
        mv_set_default(param,'k',5);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'evtol',10^8);
    
    case 'logreg'
        mv_set_default(param,'bias',100);
        mv_set_default(param,'correct_bias', true);
        mv_set_default(param,'lambda',1);
        mv_set_default(param,'max_iter',400);
        mv_set_default(param,'tolerance',1e-6);
        mv_set_default(param,'k',5);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'predict_regularisation_path',1);
        mv_set_default(param,'polyorder',3);
        
    case 'svm'
        mv_set_default(param,'bias','auto');
        mv_set_default(param,'c','auto');
        mv_set_default(param,'kernel','linear'); % 'poly' 'rbf'
        mv_set_default(param,'kernel_matrix',[]); % 'poly' 'rbf'
        mv_set_default(param,'prob',0);
        mv_set_default(param,'regularise_kernel',10e-10);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'k',3);
        mv_set_default(param,'tolerance',0.1);
        mv_set_default(param,'shrinkage_multiplier',1);
        mv_set_default(param,'q',[]);
        
        % parameters for specific kernels
        mv_set_default(param,'gamma','auto'); % RBF and polynomial kernel regularisation parameter
        mv_set_default(param,'coef0',1);    % polynomial kernel
        mv_set_default(param,'degree',2);   % degree of polynomial kernel
        
        
    case 'svm_sgd'
        mv_set_default(param,'bias',1);
        mv_set_default(param,'lambda','auto');
        mv_set_default(param,'kernel','linear');
        mv_set_default(param,'n_epochs','auto');
        mv_set_default(param,'plot',0);
        mv_set_default(param,'k',5);
        
        % parameters for specific kernels
        mv_set_default(param,'gamma',1); % RBF regularisation parameter
        
    case 'linear_svm'
        mv_set_default(param,'zscore',0);
        mv_set_default(param,'bias',1);
        mv_set_default(param,'lambda',1);
        mv_set_default(param,'max_iter',400);
        mv_set_default(param,'tolerance',1e-8);
        mv_set_default(param,'k',5);
        mv_set_default(param,'plot',0);
        mv_set_default(param,'predict_regularisation_path',1);
        mv_set_default(param,'polyorder',2);

        mv_set_default(param,'z1',0.5);
        
        % We set z2 automatically such that the [z1,z2] interval is
        % centered on 1. This leads to the polynomial interpolation being
        % of degree 4 instead of 5.
        param.z2 = 2 - param.z1; 
        
        %%% Create version of hinge loss with a polynomial interpolation 
        % in the interval z1 < 1 < z2. We need to calculate the weights
        % (the a's) for the polynomial terms.
        % It is more efficient to do this here than in train_svm since it
        % just needs to be done once. If you want to change z1 and z2 and
        % recalculate the spline parameters a, re-call
        % mv_classifier_defaults with param after setting the z1 and z2
        % fields to new values
        z1 = param.z1;
        z2 = param.z2;
        % Devise system of linear equations with the following equations
        % (left side corresponds to B, right side corresponds to y)
        % p(z1)   = 1-z1
        % p(z2)   = 0
        % p'(z1)  = -1
        % p'(z2)  = 0
        % p''(z1) = 0
        % p''(z2) = 0
        B = [1*z1^5, 1*z1^4, 1*z1^3, 1*z1^2, 1*z1, 1;
            1*z2^5, 1*z2^4, 1*z2^3, 1*z2^2, 1*z2, 1;
            5*z1^4, 4*z1^3, 3*z1^2, 2*z1^1, 1, 0;
            5*z2^4, 4*z2^3, 3*z2^2, 2*z2^1, 1, 0;
            20*z1^3, 12*z1^2, 6*z1^1, 2, 0, 0;
            20*z2^3, 12*z2^2, 6*z2^1, 2, 0, 0];
        y = [1-z1, 0, -1, 0, 0 ,0]';
        param.poly = B\y;
        % The multipler for the z^5 term is zero, since we forced z2=2-z1,
        % so we can remove it and continue with a 4-th order polynomial
        param.poly = param.poly(2:end);
        param.poly(abs(param.poly) < 10^-12) = 0;   % threshold very small values probably stemming from rounding errors to 0

        % First derivative [3rd order polynomial]
        param.d1_poly = (4:-1:1)' .* param.poly(1:end-1);
        
        % Second derivative [2nd order polynomial]
        param.d2_poly = (3:-1:1)' .* param.d1_poly(1:end-1);
        
%         % when the interval is symmetric about 1, ie z2 = 2-z1
%         syms a1 a2 a3 a4 a5 a6 z1 z2
%         B = [1*z1^5, 1*z1^4, 1*z1^3, 1*z1^2, 1*z1, 1;
%             1*(2-z1)^5, 1*(2-z1)^4, 1*(2-z1)^3, 1*(2-z1)^2, 1*(2-z1), 1;
%             5*z1^4, 4*z1^3, 3*z1^2, 2*z1^1, 1, 0;
%             5*(2-z1)^4, 4*(2-z1)^3, 3*(2-z1)^2, 2*(2-z1)^1, 1, 0;
%             20*z1^3, 12*z1^2, 6*z1^1, 2, 0, 0;
%             20*(2-z1)^3, 12*(2-z1)^2, 6*(2-z1)^1, 2, 0, 0];
%         a = linsolve(B,y);

%         % hinge-spline function value
%         h = a(1)*z^5 + a(2)*z^4 + a(3)*z^3 + a(4)*z^2 + a(5)*z + a(6);
%         % first derivative
%         dh = 5*a(1)*z^4 + 4*a(2)*z^3 + 3*a(3)*z^2 + 2*a(4)*z + a(5);
%         % Roots of h''(z)
%         ddh = 20*a(1)*z^3 + 12*a(2)*z^2 + 6*a(3)*z + 2*a(4);
   
        mv_set_default(param,'optim',optimoptions('fsolve','Algorithm',...
            'trust-region-dogleg',...
            'SpecifyObjectiveGradient',true,'Display','none') );
        
    case 'libsvm'
        mv_set_default(param,'svm_type',0);
        mv_set_default(param,'kernel_type',2);
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
        mv_set_default(param,'kernel_matrix',[]);
        
    case 'liblinear'
        mv_set_default(param,'type',1);
        mv_set_default(param,'cost',1);
        mv_set_default(param,'epsilon',0.1);
        mv_set_default(param,'eps',[]);  % use the defaults in LIBLINEAR
        mv_set_default(param,'bias',-1);
        mv_set_default(param,'weight',[]);
        mv_set_default(param,'cv',[]);
        mv_set_default(param,'c',[]);
        mv_set_default(param,'quiet',1);
        
    case 'logreg_matlab'
        mv_set_default(param,'alpha',0.01);
        mv_set_default(param,'numLambda',100);
        mv_set_default(param,'k',5);
        mv_set_default(param,'nameval',{});

    case 'ensemble'
        mv_set_default(param,'learner','lda');
        mv_set_default(param,'learner_param',[]);
        mv_set_default(param,'nsamples', 0.5);
        mv_set_default(param,'nfeatures', 0.2);
        mv_set_default(param,'nlearners', 500);
        mv_set_default(param,'stratify', 1);
        mv_set_default(param,'bootstrap', 1);
        mv_set_default(param,'strategy', 'vote');
        mv_set_default(param,'simplify', false);
        
    case 'multiclass_lda'
        mv_set_default(param,'reg','shrink');
        mv_set_default(param,'lambda','auto');
        
    case 'kernel_fda'
        mv_set_default(param,'reg','shrink');
        mv_set_default(param,'lambda',10e-5);
        mv_set_default(param,'kernel','linear');
%         mv_set_default(param,'kernel_regularisation',10e-10);
        mv_set_default(param,'kernel_matrix',[]);
        
        % parameters for specific kernels
        mv_set_default(param,'gamma','auto'); % RBF and polynomial kernel regularisation parameter
        mv_set_default(param,'coef0',1);    % polynomial kernel
        mv_set_default(param,'degree',3);   % degree of polynomial kernel
     
    otherwise, error('Unknown classifier ''%s''',classifier)
end