function cf = train_logreg(cfg,X,clabel)
% Trains a logistic regression classifier with L2 regularisation. 
%
% NOTE: Due to the exponential term in the cost function, it is recommended 
% that X (the data) is z-scored to reduce the probability of numerical
% issues due to round-off errors.
%
% Usage:
% cf = train_logreg(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% cfg          - struct with hyperparameters:
% .lambda        - regularisation hyperparameter controlling the magnitude
%                  of regularisation. If a single value is given, it is
%                  used for regularisation. If a vector of values is given,
%                  5-fold cross-validation is used to test all the values
%                  in the vector and the best one is selected
%                  Note: lambda is reciprocally related to the cost
%                  parameter C used in LIBSVM/LIBLINEAR, ie C = 1/lambda
%                  roughly
% .prob          - if 1, decision values are returned as probabilities. If
%                  0, the decision values are simply the distance to the
%                  hyperplane (default 0)
% .correct_bias  - if the number of samples in the two classes is not the
%                  same, logistic regression is biased towards the majority
%                  class. If correct_bias is 1, this is corrected for by
%                  adjusting the bias term
%
% Further parameters (that usually do not need to be changed):
% bias          - if >0 augments the data with a bias term equal to the
%                 value of bias:  X <- [X; bias], and augments the weight
%                 vector with the bias variable w <- [w; b].
%                 If 0, the bias is assumed to be 0. By default, it is set
%                 to a large value (bias=100). This prevents that b is
%                 penalised by the regularisation term.
% k             - the number of folds in the k-fold cross-validation for
%                 the lambda search
% plot          - if a lambda search is performed, produces diagnostic
%                 plots including the regularisation path and
%                 cross-validated accuracy as a function of lambda
%
% BACKGROUND:
% Logistic regression introduces a non-linearity over the linear regression
% term f(x) = w * x + b by means of the sigmoid function s(x) = 1/(1+e^-x),
% hence:       s(f(x)) = 1 / ( 1 + e^-f(x) )
% and fits the sigmoid function to the data. Logistic regression is a
% linear function of the log-odds and directly models class probabilities.
% The log likelihood function
% including a L2 regularisation term can be arranged as
%
%      L(w,lambda) = SUM log(1+exp(-yi*w*xi)) + lambda * ||w||^2
%
% where w is the coefficient vector and lambda is the regularisation
% strength, yi = {-1,+1} are the class labels, and xi the samples. This is
% a convex optimisation problem that can be solved by unconstrained
% minimisation.
%
% IMPLEMENTATION DETAILS:
% A Trust Region Dogleg algorithm (TrustRegionDoglegGN.m) is used to
% optimise w. 
% Hyperparameter optimisation is very costly, since the classifier
% has to be trained for each value of the hyperparameter. To reduce the
% number of iterations, warm starts are used wherein the next w is
% initialised (w_init) as follows:
% - in iteration 1, w_init is initialised as the zero vector
% - in iterations 2 and 3, w_init is initialised by the previous w's (w_1
%   and w_2)
% - in iterations 4+, if predict_regularisation_path=1, then w_init is 
%   initialised by a predicted w: for the
%   k-th iteration, a quadratic polynomial is fit through w_k-2, w_k-1 and
%   wk as a function of lambda. The polynomial is then evaluated at
%   lambda_k to obtain the prediction. Simulations show that this approach
%   substantially uses the number of to convergence and hence computation
%   time
%
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold
%
%References:
% Lin, Weng & Keerthi (2008). Trust Region Newton Method for Large-Scale 
% Logistic Regression. Journal of Machine Learning Research, 9, 627-650

% (c) Matthias Treder 2017

[N, nFeat] = size(X);
X0 = X;

cf = [];

% Make sure labels come as column vector
clabel = double(clabel(:));

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

% Stack labels in diagonal matrix for matrix multiplication during
% optimisation
Y = diag(clabel);

% Take vector connecting the class means as initial guess [it does not seem
% to speed up convergence though so we keep w0 = 0 for now]
% w0 = double(mean(X0(clabel==1,:))) -  double(mean(X0(clabel==-1,:)));
% w0 = w0' / norm(w0);

% Augment X with bias
if cfg.bias > 0
    X0 = cat(2, X0, ones(N,1) * cfg.bias );
    nFeat = nFeat + 1;
end

I = eye(nFeat);

% Initialise w with zeros 
w0 = zeros(nFeat,1);

% logfun = @(w) lr_objective_tanh(w);
% logfun = @(w) lr_objective(w);
% logfun = @(w) lr_gradient(w);
logfun = @(w) lr_gradient_and_hessian_tanh(w);

%% Automatic regularisation
if ischar(cfg.lambda) && strcmp(cfg.lambda,'auto')
    cfg.lambda = logspace(-4,3,10);
end

%% Find best lambda using (nested) cross-validation
if numel(cfg.lambda)>1
    % Perform inner cross validation by again partitioning the training
    % data into folds. Then, cycle through all the lambda's, calculate
    % the classifier, and validate it on the test set. The lambda giving
    % the best result is then taken forward and a model is trained on the
    % full data using the best lambda.
    
    CV = cvpartition(N,'KFold',cfg.k);
    ws = zeros(nFeat, numel(cfg.lambda));
    acc = zeros(numel(cfg.lambda),1);
    
    if cfg.plot
        C = zeros(numel(cfg.lambda));
        iter_tmp = zeros(numel(cfg.lambda),1);
        delta_tmp = zeros(numel(cfg.lambda),1);
        iter = zeros(numel(cfg.lambda),1);
        delta = zeros(numel(cfg.lambda),1);
        wspred= ws;
    end
    
    if cfg.predict_regularisation_path
        % Create predictor matrix for the polynomial approximation 
        % of the regularisation path by taking the lambda's to the powers
        % up to the polyorder.
        polyvec = 0:cfg.polyorder;
        % Use the log of the lambda's to get a better conditioned matrix
        qpred = cell2mat( arrayfun(@(n) log(cfg.lambda(:)).^n, polyvec,'Un',0));
    end
    
    % --- Start cross-validation ---
    for ff=1:cfg.k
        % Training data
        X = X0(CV.training(ff),:);
        YX = Y(CV.training(ff),CV.training(ff))*X;
        
        % Sum of samples needed for the gradient
        sumyxN = sum(YX)'/N;
        
        % --- Loop through lambdas ---
        for ll=1:numel(cfg.lambda)
            lambda = cfg.lambda(ll);
            
            % Warm-starting the initial w: wstart
            if ll==1
                wstart = w0;
            elseif cfg.predict_regularisation_path ...
                    && ll>cfg.polyorder+1      % make sure that enough terms have been calculated
                % Fit polynomial to regularisation path
                % and predict next w(lambda_k)
                quad = qpred(ll-cfg.polyorder-1:ll-1,:)\(ws(:,ll-cfg.polyorder-1:ll-1)');
                wstart = ( repmat(log(lambda),[1,numel(polyvec)]).^polyvec * quad)';
            else
                % Use the result obtained in the previous step lambda_k-1
                wstart = ws(:,ll-1);
            end
            if cfg.plot
                wspred(:,ll)= wstart;
                [ws(:,ll),iter_tmp(ll),delta(ll)] = TrustRegionDoglegGN(logfun, wstart, cfg.tolerance, cfg.max_iter,ll);
            else
                ws(:,ll) = TrustRegionDoglegGN(logfun, wstart, cfg.tolerance, cfg.max_iter,ll);
            end
        end
        if cfg.plot
            delta = delta + delta_tmp;
            iter = iter + iter_tmp;
            C = C + corr(ws);
        end
        
        cl = clabel(CV.test(ff));
        % Calculate classification accuracy by multiplying decision values
        % with the class label
        acc = acc + sum( (X0(CV.test(ff),:) * ws) .* repmat(cl(:),[1,numel(cfg.lambda)]) > 0)' / CV.TestSize(ff);
    end
    
    acc = acc / cfg.k;
    
    [~, best_idx] = max(acc);
    
    % Diagnostic plots if requested
    if cfg.plot
        figure,
        nCol=3; nRow=1;
        subplot(nRow,nCol,1),imagesc(C); title({'Mean correlation' 'between w''s'}),xlabel('lambda#')
        subplot(nRow,nCol,2),plot(delta),title({'Mean trust region' 'size at termination'}),xlabel('lambda#')
        subplot(nRow,nCol,3),plot(iter/cfg.k),hold all,
        title({'Mean number' 'of iterations (across folds)'}),xlabel('lambda#')
        
        % Plot regularisation path (for the last training fold)
        figure
        for ii=1:nFeat, semilogx(cfg.lambda,ws(ii,:),'o-','MarkerFaceColor','w'), hold all, end
        plot(xlim,[0,0],'k-'),title('Regularisation path for last iteration'),xlabel('lambda#')
        
        % Plot cross-validated classification performance
        figure
        semilogx(cfg.lambda,acc)
        title([num2str(cfg.k) '-fold cross-validation performance'])
        hold all
        plot([cfg.lambda(best_idx), cfg.lambda(best_idx)],ylim,'r--'),plot(cfg.lambda(best_idx), acc(best_idx),'ro')
        xlabel('Lambda'),ylabel('Accuracy'),grid on
        
        % Plot first two dimensions
        figure
        plot(ws(1,:),ws(end,:),'ko-')
        hold all, plot(wspred(1,2:end),wspred(end,2:end),'+')
        title('Regularisation path for 2 features')
        legend({'w' 'predicted w'})
        
    end
else
    % there is just one lambda: no grid search
    best_idx = 1;
end

lambda = cfg.lambda(best_idx);

%% Train classifier on the full training data (using the best lambda)
YX = Y*X0;
sumyxN = sum(YX)'/N;
X = X0;

w = TrustRegionDoglegGN(logfun, w0, cfg.tolerance, cfg.max_iter, 1);

%% Set up classifier struct
if cfg.bias > 0
    cf.w = w(1:end-1);
    cf.b = w(end);
    
    % Bias term needs correct scaling 
    cf.b = cf.b * cfg.bias;

    if cfg.correct_bias
        % Correct the bias term such that it is in between the class means
        o = X(:, 1:end-1) * cf.w;
        cf.b = - ( mean(o(clabel==1)) + mean(o(clabel==-1)) )/2;
    end
else
    cf.w = w;
    cf.b = 0;
end
cf.lambda = lambda;



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

%     function [f,g,h] = lr_objective(w)
%         % Evaluate exponential for all samples
%         sigma = logreg_fix(YX*w); %1./(1+exp(-YX*w));
%         
%         % Function value (loss)
%         f = sum(-log(sigma))/N + lambda * 0.5 * (w'*w);
%         
%         % Gradient
%         if nargout>1
%             g = YX'*sigma/N - sumyxN + lambda * w;
%         end
%         
%         % Hessian
%         if nargout>2
%             h = lambda * I + (X' .* (sigma .* (1 - sigma))') * X/N;
%         end
%     end
% 
%     function [f,g,H] = lr_objective_tanh(w)
%         % Evaluate exponential for all samples
%         sigma = 0.5 + 0.5 * tanh(YX*w/2);
%         
%         % Function value (loss)
%         f = sum(-log(sigma))/N + lambda * 0.5 * (w'*w);
%         
%         % Gradient
%         if nargout>1
%             g = YX'*sigma/N - sumyxN + lambda * w;
%         end
%         
%         % Hessian
%         if nargout>2
%             H = lambda * I + (X'.*(sigma .* (1 - sigma))') * X/N;
%         end
%     end

%     function [g,h] = lr_gradient(w)
%         % Logistic gradient and Hessian
%         
%         sigma = logreg_fix(YX*w);
%         
%         % Gradient
%         g = YX'*sigma/N - sumyxN + lambda * w;
%         
%         % Hessian of loss function (serves here as gradient)
%         if nargout>1
%             h = lambda * I + (X .* (sigma .* (1 - sigma)))' * X/N;  % faster
%         end
%     end

    function [g,h] = lr_gradient_and_hessian_tanh(w)
        % Logistic gradient and Hessian expressed using the hyperbolic 
        % tangent
        % Using the identity: 1 / (1 + exp(-x)) = 0.5 + 0.5 * tanh(x/2)
        sigma = 0.5 + 0.5 * tanh(YX*w/2);
        
        % Gradient
        g = (sigma' * YX)'/N - sumyxN + lambda * w;
        
        % Hessian
        if nargout>1
            h = lambda * I + bsxfun(@times, X, sigma .* (1 - sigma))' * X/N;  % faster to first multiply X by sigma(1-sigma)
        end
    end

%     function xo = log_logreg_fix(x)
%         % This is a fix to the LOG logistic loss function found on
%         % http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
%         % also used in the LIBLINEAR code. It prevents exp from overflow.
%         xo = x;
%         xo(x>=0) = log(1+exp(-x(x>=0)));
%         xo(x<0)  = log(1+exp(x(x<0))) - x(x<0);
%     end

    function xo = logreg_fix(x)
        % This is a fix to the logistic loss function found on
        % http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        % and used in the LIBLINEAR code. It prevents exp from overflowing.
        % Logistic loss: 1./1+exp(-YX*w);
        xo = x;
        xo(x>=0) = 1./(1+exp(-x(x>=0)));
        xo(x<0)  = exp(x(x<0))./(1+exp(x(x<0)));
    end
end