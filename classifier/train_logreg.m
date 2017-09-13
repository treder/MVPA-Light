function [cf, b, stats] = train_logreg(cfg,X,clabel)
% Trains a logistic regression classifier. To avoid overfitting, L2
% regularisation is used. Requires the Optimisation toolbox.
%
% Logistic regression introduces a non-linearity over the linear regression
% term f(x) = w * x + b by means of the sigmoid function s(x) = 1/(1+e^-x),
% hence:       s(f(x)) = 1 / ( 1 + e^-f(x) )
% and fits the sigmoid function to the data. The log likelihood function
% including a L2 regularisation term can be arranged as
%
%      L(w,lambda) = SUM log(1+exp(-yi*w*xi)) + lambda * ||w||^2
%
% where w is the coefficient vector and lambda is the regularisation
% strength, yi = {-1,+1} are the class labels, and xi the samples. This is
% a convex optimisation problem that is solved by unconstrained
% minimisation using Matlab's fsolve.
%
% 
%
% Usage:
% cf = train_logreg(cfg,X,clabel)
% 
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing 
%                  1's (class 1) and 2's (class 2)
%
% cfg          - struct with hyperparameters:
% normalise      - zscores the training data. Since the loss function
%                  involves exp, this assures that the data behaves nicely.
%                  Note that the test set should be zscored as well.
%                  Ideally, you zscore the full data before starting
%                  classification, then normalise can be 0 (default 0)
% intercept      - augments the data with an intercept term (recommended)
%                  (default 1). If 0, the intercept is assumed to be 0
% lambda         - regularisation hyperparameter controlling the magnitude
%                  of regularisation. If a single value is given, it is
%                  used for regularisation. If a vector of values is given,
%                  5-fold cross-validation is used to test all the values
%                  in the vector and the best one is selected 
%                  (default  ???)
%                  

%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold 
%

% Reference:
% RE Fan, KW Chang, CJ Hsieh, XR Wang, CJ Lin (2008).
% LIBLINEAR: A library for large linear classification
% Journal of machine learning research 9 (Aug), 1871-1874

% Matthias Treder 2017

X= double(X);
lambda = cfg.lambda;
[N, nFeat] = size(X);

if cfg.normalise
    X = zscore(X);
end

% Need class labels 1 and -1
clabel(clabel == 2) = -1;

% Stack labels in diagonal matrix for matrix multiplication during
% optimisation
Y = diag(clabel); 

% Take vector connecting the class means as initial guess for speeding up
% convergence
w0 = double(mean(X(clabel==1,:))) -  double(mean(X(clabel==-1,:)));
w0 = w0' / norm(w0);

% Augment X with intercept
if cfg.intercept
    X = cat(2,X, ones(N,1));
    w0 = [w0; 0];
    nFeat = nFeat + 1;
end

YX = Y*X;

% Sum of samples needed for the gradient
sumx = sum(YX)';

I = eye(nFeat);

% fminunc settings
%  opt_unc = optimoptions('fminunc','Algorithm','trust-region',...
%     'SpecifyObjectiveGradient',true,'Display','none',...
%     'HessianFcn','objective');
% w = fminunc(@(w) lr_objective4fminunc(w),wran, opt_unc);

w = fsolve(@(w) lr_objective4fsolve(w), w0, cfg.optim);

cf = [];
cf.w = w(1:end-1);
cf.b = w(end);



%%% Logistic regression objective function. Given w, data X and
%%% regularisation parameter lambda, returns 
%%% f: function value at point w
%%% g: value of gradient at point w
%%% h: value of the Hessian at point w
%%%
%%% Based on formulas (2)-(4) in:
%%% Lin C Weng R Keerthi S (2007). Trust region Newton methods for 
%%% large-scale logistic regression. Proceedings of the 24th international 
%%% conference on Machine learning - ICML '07. pp: 561-568

    function [f,g,h] = lr_objective4fminunc(w)
        % Evaluate exponential for all samples
        s = logreg_fix(YX*w); %1+exp(-YX*w);
        logs = log_logreg_fix(YX*w) ; %log(s);
        
        % Function value (loss)
        f = sum(logs) + lambda * 0.5 * (w'*w);
        
        % Gradient
        if nargout>1
            g = YX'*s - sumx + lambda * w;
%             g = YX'*(1./s) - sumx + lambda * w;
        end
        
%         % Hessian
        if nargout>2
%             D = 1./s - 1./(s.^2);
            D = s .* (1 - s);
            h = lambda * I + X' * diag(D) * X;
        end
    end

    function [g,h] = lr_objective4fsolve(w)
        % Directly provide the gradient and solve for zeros
%         s = 1+exp(YX*w);
%         sig = 1./(1+exp(-YX*w));
        
        % Gradient of loss function (serves here as function value)
%         g = X'*(1./s) - sumx + lambda * w;
%         g = lambda * w + X' * ( sig - 1);
%         g = w + lambda * YX' * ( sig - 1);

        s = logreg_fix(YX*w);
        g = YX'*s - sumx + lambda * w;
        
        % Hessian of loss function (serves here as gradient)
        if nargout>1
            D = s .* (1 - s);
            h = lambda * I + X' * diag(D) * X;
        end
    end

    function xo = log_logreg_fix(x)
        % This is a fix to the log logistic loss function found on 
        % http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        % and apparently used in the LIBLINEAR code. It prevents exp from
        % overflowing
        xo = x;
        xo(x>=0) = log(1+exp(-x(x>=0)));
        xo(x<0)  = log(1+exp(x(x<0))) - x(x<0);
    end

    function xo = logreg_fix(x)
        % This is a fix to the logistic loss function found on 
        % http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        % and apparently used in the LIBLINEAR code. It prevents exp from
        % overflowing
        xo = x;
        xo(x>=0) = 1./(1+exp(-x(x>=0)));
        xo(x<0)  = exp(x(x<0))./(1+exp(x(x<0)));
    end
end