function cf = train_logreg(cfg,X,clabel)
% Trains a logistic regression classifier with L2 regularisation. It is
% recommended that X (the data) is z-scored to avoid numerical issues for
% optimisation.
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
% zscore         - zscores the training data. The scaling will be saved in 
%                  the classifier and applied to the test set (in test_logreg). 
%                  This option is not required if the data has been
%                  z-scored already (default 0)
% intercept      - augments the data with an intercept term (recommended)
%                  (default 1). If 0, the intercept is assumed to be 0
% lambda         - regularisation hyperparameter controlling the magnitude
%                  of regularisation. If a single value is given, it is
%                  used for regularisation. If a vector of values is given,
%                  5-fold cross-validation is used to test all the values
%                  in the vector and the best one is selected 
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
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold 

% (c) Matthias Treder 2017

X= double(X);
[N, nFeat] = size(X);
X0 = X;

lambda = cfg.lambda;
cf = [];

if cfg.zscore
    cf.zscore = 1;
    [X0,cf.mean,cf.std] = zscore(X0);
else
    cf.zscore = 0;
end

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

% Stack labels in diagonal matrix for matrix multiplication during
% optimisation
Y = diag(clabel); 

% Take vector connecting the class means as initial guess for speeding up
% convergence
w0 = double(mean(X0(clabel==1,:))) -  double(mean(X0(clabel==-1,:)));
w0 = w0' / norm(w0);

% Augment X with intercept
if cfg.intercept
    X0 = cat(2,X0, ones(N,1));
    w0 = [w0; 0];
    nFeat = nFeat + 1;
end

I = eye(nFeat);

% cfg.lambda = logspace(-10,2,50);

%% FSOLVE
% fun = @(w) lr_gradient_tanh(w);
% 
% X = X0;
% YX = Y*X;
% sumyx = sum(YX)';
% 
% lambda = 1;
% lambda = 5.3367e-05;
% 
% [w_fsolve,~,~,stat] = fsolve(@(w) lr_gradient(w), w0, cfg.optim);
% 
% [w,iter] = TrustRegionDoglegGN(fun, w0, cfg.tolerance, cfg.max_iter, 1);

%% FSOLVE - 5-fold CV
% K = 5;
% CV = cvpartition(N,'KFold',K);
% ws_fsolve = zeros(nFeat, numel(cfg.lambda));
% fun = @(w) lr_gradient_tanh(w);
% 
% tic
% for ff=1:K
%     X = X0(CV.training(ff),:);
%     YX = Y(CV.training(ff),CV.training(ff))*X;
% 
%     % Sum of samples needed for the gradient
%     sumyx = sum(YX)';
% 
%     for ll=1:numel(cfg.lambda)
%         lambda = cfg.lambda(ll);
%         if ll==1
%             ws_fsolve(:,ll) = fsolve(@(w) lr_gradient(w), w0, cfg.optim);
%         else
%             ws_fsolve(:,ll) = fsolve(@(w) lr_gradient(w), ws_fsolve(:,ll-1), cfg.optim);
%         end
%         
%     end
% end
% toc


% fun = @(w) lr_objective_tanh(w);
% fun = @(w) lr_objective(w);
% fun = @(w) lr_gradient_tanh(w);
fun = @(w) lr_gradient(w);

%% Find best lambda using cross-validation
if numel(cfg.lambda)>1
    CV = cvpartition(N,'KFold',cfg.K);
    ws = zeros(nFeat, numel(cfg.lambda));
    acc = zeros(numel(cfg.lambda),1);

    if cfg.plot
        C = zeros(numel(cfg.lambda));
        iter_tmp = zeros(numel(cfg.lambda),1);
        delta_tmp = zeros(numel(cfg.lambda),1);
        iter = zeros(numel(cfg.lambda),1);
        delta = zeros(numel(cfg.lambda),1);
    end
    
    for ff=1:cfg.K
        X = X0(CV.training(ff),:);
        YX = Y(CV.training(ff),CV.training(ff))*X;
        
        % Sum of samples needed for the gradient
        sumyx = sum(YX)';
        
        for ll=1:numel(cfg.lambda)
            lambda = cfg.lambda(ll);
            if cfg.plot
                % Need to acquire diagnostic information as well
                if ll==1
                    [ws(:,ll),iter_tmp(ll),delta(ll)] = TrustRegionDoglegGN(fun, w0, cfg.tolerance, cfg.max_iter,ll);
                else
                    [ws(:,ll),iter_tmp(ll),delta(ll)] = TrustRegionDoglegGN(fun, ws(:,ll-1), cfg.tolerance, cfg.max_iter,ll);
                end
            else
                if ll==1
                    ws(:,ll) = TrustRegionDoglegGN(fun, w0, cfg.tolerance, cfg.max_iter,ll);
                else
                    ws(:,ll) = TrustRegionDoglegGN(fun, ws(:,ll-1), cfg.tolerance, cfg.max_iter,ll);
                end
            end
        end
        if cfg.plot
            delta = delta + delta_tmp;
            iter = iter + iter_tmp;
        end
        
        cl = clabel(CV.test(ff));
        acc = acc + sum( (X0(CV.test(ff),:) * ws) .* cl(:) > 0)' / CV.TestSize(ff);
        if cfg.plot
            C = C + corr(ws);
        end
    end
    
    acc = acc / cfg.K;
    
    [val, idx] = max(acc);
    lambda = cfg.lambda(idx);
    
    % Diagnostic plots if requested
    if cfg.plot
        figure,
        nCol=3; nRow=1;
        subplot(nRow,nCol,1),imagesc(C); title({'Mean correlation' 'between w''s'}),xlabel('lambda#')
        subplot(nRow,nCol,2),plot(delta),title({'Mean trust region' 'size at termination'}),xlabel('lambda#')
        subplot(nRow,nCol,3),plot(iter_tmp),title({'Mean number' 'of iterations'}),xlabel('lambda#')
        
        % Plot regularisation path (for the last training fold)
        figure
        for ii=1:nFeat, semilogx(cfg.lambda,ws(ii,:),'-'), hold all, end
        plot(xlim,[0,0],'k-'),title('Regularisation path for last iteration'),xlabel('lambda#')
        
        % Plot cross-validated classification performance
        figure
        semilogx(cfg.lambda,acc)
        title([num2str(cfg.K) '-fold cross-validation performance'])
        hold all
        plot([lambda, lambda],ylim,'r--'),plot(lambda, acc(idx),'ro')
        xlabel('Lambda'),ylabel('Accuracy')
    end
end

%% Train classifier
YX = Y*X0;
sumyx = sum(YX)';
X = X0;

% fun = @(w) lr_objective_tanh(w);
% fun = @(w) lr_objective(w);
fun = @(w) lr_gradient_tanh(w);
% fun = @(w) lr_gradient(w);

w = TrustRegionDoglegGN(fun, w0, cfg.tolerance, cfg.max_iter, 1);

%% Set up classifier
if cfg.intercept
    cf.w = w(1:end-1);
    cf.b = w(end);
else
    cf.w = w;
    cf.b = 0;
end

%%
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

    function [f,g,h] = lr_objective(w)
        % Evaluate exponential for all samples
        sigma = logreg_fix(YX*w); %1./(1+exp(-YX*w));
        
        % Function value (loss)
        f = sum(-log(sigma)) + lambda * 0.5 * (w'*w);
        
        % Gradient
        if nargout>1
            g = YX'*sigma - sumyx + lambda * w;
        end
        
        % Hessian
        if nargout>2
            h = lambda * I + (X' .* (sigma .* (1 - sigma))') * X;
        end
    end

    function [f,g,H] = lr_objective_tanh(w)
        % Evaluate exponential for all samples
        sigma = 0.5 + 0.5 * tanh(YX*w/2);
        
        % Function value (loss)
        f = sum(-log(sigma)) + lambda * 0.5 * (w'*w);
        
        % Gradient
        if nargout>1
            g = YX'*sigma - sumyx + lambda * w;
        end
        
        % Hessian
        if nargout>2
            H = lambda * I + (X'.*(sigma .* (1 - sigma))') * X;
        end
    end


    function [g,h] = lr_gradient(w)
        % Directly provide the gradient and solve for zeros
%         sigma = 1./1+exp(-YX*w);
        sigma = logreg_fix(YX*w);
        
        % Gradient
        g = YX'*sigma - sumyx + lambda * w;
        
        % Hessian of loss function (serves here as gradient)
        if nargout>1
            h = lambda * I + (X .* (sigma .* (1 - sigma)))' * X;  % faster
        end
    end

    function [g,h] = lr_gradient_tanh(w)
        % Logistic function expressed using the hyperbolic tangent
        % 1 / (1 + exp(-x)) = 1/2 + 1/2 * tanh(x/2)
        sigma = 0.5 + 0.5 * tanh(YX*w/2);
        
        % Gradient
        g = YX'*sigma - sumyx + lambda * w;
        
        % Hessian of loss function (serves here as gradient)
        if nargout>1
            h = lambda * I + (X .*(sigma .* (1 - sigma)))' * X;  % faster
        end
    end

    function xo = log_logreg_fix(x)
        % This is a fix to the LOG logistic loss function found on 
        % http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        % and allegedly used in the LIBLINEAR code. It prevents exp from
        % overflowing
        xo = x;
        xo(x>=0) = log(1+exp(-x(x>=0)));
        xo(x<0)  = log(1+exp(x(x<0))) - x(x<0);
    end

    function xo = logreg_fix(x)
        % This is a fix to the logistic loss function found on 
        % http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        % and allegedly used in the LIBLINEAR code. It prevents exp from
        % overflowing
        xo = x;
        xo(x>=0) = 1./(1+exp(-x(x>=0)));
        xo(x<0)  = exp(x(x<0))./(1+exp(x(x<0)));
    end
end