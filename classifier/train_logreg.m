<<<<<<< HEAD
function cf = train_logreg(param,X,clabel)
=======
function cf = train_logreg(cfg,X,clabel)
>>>>>>> preprocess
% Trains a logistic regression classifier with logF or L2 
% regularisation. 
%
% Note: Due to the exponential term in the cost function, it is recommended 
% that X (the data) is z-scored to reduce the probability of numerical
% issues due to round-off errors.
%
% Usage:
<<<<<<< HEAD
% cf = train_logreg(param,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% param          - struct with hyperparameters:
% .reg           - regularisation approach
%                 'logf': log-F(1,1) regularisation via data augmentation
%                  'l2': an L2 regularisation term is added to the cost
%                  function. The magnitude of regularisation is controlled
%                  by the lambda hyperparameter
%                  No additional hyperparameter required (default 'logf')
% .lambda        - if reg='l2', then lambda is the 
%                  regularisation hyperparameter controlling the magnitude
=======
% cf = train_logreg(cfg,X,clabel)
%
%Parameters:
% X              - [instances x features] matrix of training instances
% clabel         - [instances x 1] vector of class labels
%
% cfg          - struct with hyperparameters:
% .reg           - regularisation approach
%                 'logf': log-F(1,1) regularisation via data augmentation
%                 (default)
%                  'l2': an L2 regularisation term is added to the cost
%                  function. The magnitude of regularisation is controlled
%                  by the lambda hyperparameter
%                 No additional hyperparameter required (default 'logf')
% .lambda        - regularisation hyperparameter controlling the magnitude
>>>>>>> preprocess
%                  of L2 regularisation. If a single value is given, it is
%                  used for regularisation. If a vector of values is given,
%                  5-fold cross-validation is used to test all the values
%                  in the vector and the best one is selected
%                  Note: lambda is reciprocally related to the cost
%                  parameter C used in LIBSVM/LIBLINEAR, ie C = 1/lambda
%                  roughly
<<<<<<< HEAD
% .correct_bias  - if the number of samples in the two classes is not
=======
% .correct_bias  - if the number of instances in the two classes is not
>>>>>>> preprocess
%                  equal, logistic regression is biased towards the majority
%                  class. If correct_bias is 1, this is corrected for by
%                  adjusting the weights (note: if the weights have
%                  already been set by the user, bias correction is applied
%                  to the user weights)
<<<<<<< HEAD
% .weights       - [samples x 1] vector of sample weights. This
%                  allows for up/down weighting of samples such that they
%                  contribute more/less to the loss function. By default,
%                  all samples are treated equally (weights all 1's)
=======
% .weights       - [instances x 1] vector of instance weights. This
%                  allows for up/down weighting of instances such that they
%                  contribute more/less to the loss function. By default,
%                  all instances are treated equally (weights all 1's)
>>>>>>> preprocess
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
<<<<<<< HEAD
% strength, yi = {-1,+1} are the class labels, and xi the samples. This is
=======
% strength, yi = {-1,+1} are the class labels, and xi the instances. This is
>>>>>>> preprocess
% a convex optimisation problem that can be solved by unconstrained
% minimisation.
%
% If the regularisation is 'logf', then a log-F(1,1) prior is imposed. This
% is realised by augmenting the data rather than applying a penalty to the ML
<<<<<<< HEAD
% estimate. A weight of 0.5 is applied to the augmented samples.
=======
% estimate. A weight of 0.5 is applied to the augmented instances.
>>>>>>> preprocess
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
% King, G., & Zeng, L. (2001). Logistic Regression in Rare Events Data. 
% Political Analysis, 9(2), 137?163. https://doi.org/10.1162/00208180152507597
%
% Lin, Weng & Keerthi (2008). Trust Region Newton Method for Large-Scale 
% Logistic Regression. Journal of Machine Learning Research, 9, 627-650
%
% Rahman, M. S., & Sultana, M. (2017). Performance of Firth-and logF-type 
% penalized methods in risk prediction for small or sparse binary data. 
% BMC Medical Research Methodology, 17(1), 33. https://doi.org/10.1186/s12874-017-0313-9

% (c) matthias treder

[N, nfeat] = size(X);
X0 = X;

% Make sure labels come as column vector
clabel = double(clabel(:));

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

% Take vector connecting the class means as initial guess [it does not seem
% to speed up convergence though so we keep w0 = 0 for now]
% w0 = double(mean(X0(clabel==1,:))) -  double(mean(X0(clabel==-1,:)));
% w0 = w0' / norm(w0);

% Augment X with bias
<<<<<<< HEAD
if param.bias > 0
    X0 = cat(2, X0, ones(N,1) * param.bias );
=======
if cfg.bias > 0
    X0 = cat(2, X0, ones(N,1) * cfg.bias );
>>>>>>> preprocess
    nfeat = nfeat + 1;
end

I = eye(nfeat);

% Initialise w with zeros 
w0 = zeros(nfeat,1);

% Initialise weights if none given
<<<<<<< HEAD
if isempty(param.weights) 
    param.weights = ones(N, 1);
end

%% log-F(1,1) regularisation
if strcmp(param.reg, 'logf')
=======
if isempty(cfg.weights) 
    cfg.weights = ones(N, 1);
end

%% log-F(1,1) regularisation
if strcmp(cfg.reg, 'logf')
>>>>>>> preprocess
    logfun = @(w) lr_gradient_and_hessian_tanh(w);
    
    % log-F(1,1) regularisation can be implemented by using the standard ML
    % estimation and instead augmenting the data in the following way:
<<<<<<< HEAD
    % - add 2*nfeatures new samples
    % - each added sample has the value 1 for a given feature, and 0's
    %   for all other features and intercept
    % - each such sample occurs twice, once with y=-1 and once y=1
=======
    % - add 2*nfeatures new instances
    % - each added instance has the value 1 for a given feature, and 0's
    %   for all other features and intercept
    % - each such instance occurs twice, once with y=-1 and once y=1
>>>>>>> preprocess
    % This assures that the data is not linearly separable and 
    % constitutes a form of regularisation.
    % see also 
    % http://prema.mf.uni-lj.si/files/2015-11%20Bordeaux%20Penalized%20likelihood%20Logreg%20rare%20events_e19.pdf
    % http://prema.mf.uni-lj.si/files/FLICFLAC_final_9e6.pdf
    
<<<<<<< HEAD
    nadd = nfeat - (param.bias > 0); % need to ignore the bias term if there is one
    augment = cat(1, eye(nadd), eye(nadd));
    if param.bias > 0
=======
    nadd = nfeat - (cfg.bias > 0); % need to ignore the bias term if there is one
    augment = cat(1, eye(nadd), eye(nadd));
    if cfg.bias > 0
>>>>>>> preprocess
        X0 = cat(1, X0, [augment, zeros(2*nadd, 1)]);
    else
        X0 = cat(1, X0, augment);
    end
    
    % each augmented observation has a weight of 0.5
<<<<<<< HEAD
    param.weights(end+1:end+2*nadd) = 0.5;
    
    % add class labels for augmented samples
=======
    cfg.weights(end+1:end+2*nadd) = 0.5;
    
    % add class labels for augmented instances
>>>>>>> preprocess
    clabel = [clabel(:); ones(nadd, 1); -1*ones(nadd, 1)];
    
    % Stack labels in diagonal matrix for matrix multiplication during
    % optimisation
    Y = diag(clabel);
    
<<<<<<< HEAD
    % Adjust N for the additional samples
=======
    % Adjust N for the additional instances
>>>>>>> preprocess
    N = numel(clabel);
end

%% Bias correction using weights
% The bias correction must come after logf regularisation
<<<<<<< HEAD
% (because extra samples with their own weights have to be added in first) 
% but before l2 regularisation (because the final weights are required
% for the hyperparameter tuning)
if param.correct_bias
=======
% (because extra instances with their own weights have to be added in first) 
% but before l2 regularisation (because the final weights are required
% for the hyperparameter tuning)
if cfg.correct_bias
>>>>>>> preprocess
    % Assume the classes in the population occur each with equal probability 0.5,
    % then use Eq (8) in King & Zeng (2001)
    tau = 0.5;
    ybar = sum(clabel==1)/N;
<<<<<<< HEAD
    param.weights(clabel== 1) = param.weights(clabel== 1) * tau/ybar;
    param.weights(clabel==-1) = param.weights(clabel==-1) * (1-tau)/(1-ybar);
end

%% Regularisation
if strcmp(param.reg, 'l2')
=======
    cfg.weights(clabel== 1) = cfg.weights(clabel== 1) * tau/ybar;
    cfg.weights(clabel==-1) = cfg.weights(clabel==-1) * (1-tau)/(1-ybar);
end

%% Regularisation
if strcmp(cfg.reg, 'l2')
>>>>>>> preprocess

    %% L2 regularisation 
    logfun = @(w) lr_gradient_and_hessian_tanh_L2(w);
    
    % We must subselect the weights for each training iteration
<<<<<<< HEAD
    all_weights = param.weights;
=======
    all_weights = cfg.weights;
>>>>>>> preprocess
    
    % Stack labels in diagonal matrix for matrix multiplication during
    % optimisation
    Y = diag(clabel);
    
    %%% Searchgrid for lambda
<<<<<<< HEAD
    if ischar(param.lambda) && strcmp(param.lambda,'auto')
        param.lambda = logspace(-4,3,10);
    end
    
    %%% Find best lambda using (nested) cross-validation
    if numel(param.lambda)>1
=======
    if ischar(cfg.lambda) && strcmp(cfg.lambda,'auto')
        cfg.lambda = logspace(-4,3,10);
    end
    
    %%% Find best lambda using (nested) cross-validation
    if numel(cfg.lambda)>1
>>>>>>> preprocess
        % Perform inner cross validation by again partitioning the training
        % data into folds. Then, cycle through all the lambda's, calculate
        % the classifier, and validate it on the test set. The lambda giving
        % the best result is then taken forward and a model is trained on the
        % full data using the best lambda.
        
<<<<<<< HEAD
        CV = cvpartition(N,'KFold',param.k);
        ws = zeros(nfeat, numel(param.lambda));
        acc = zeros(numel(param.lambda),1);
        
        if param.plot
            C = zeros(numel(param.lambda));
            iter_tmp = zeros(numel(param.lambda),1);
            delta_tmp = zeros(numel(param.lambda),1);
            iter = zeros(numel(param.lambda),1);
            delta = zeros(numel(param.lambda),1);
            wspred= ws;
        end
        
        if param.predict_regularisation_path
            % Create predictor matrix for the polynomial approximation
            % of the regularisation path by taking the lambda's to the powers
            % up to the polyorder.
            polyvec = 0:param.polyorder;
            % Use the log of the lambda's to get a better conditioned matrix
            qpred = cell2mat( arrayfun(@(n) log(param.lambda(:)).^n, polyvec,'Un',0));
        end
        
        % --- Start cross-validation ---
        for ff=1:param.k
            % Training data
            X = X0(CV.training(ff),:);
            YX = Y(CV.training(ff),CV.training(ff))*X;
            param.weights = all_weights(CV.training(ff));
             
            % Sum of samples needed for the gradient
            sumyxN = sum(YX)'/N;
            
            % --- Loop through lambdas ---
            for ll=1:numel(param.lambda)
                lambda = param.lambda(ll);
=======
        CV = cvpartition(N,'KFold',cfg.k);
        ws = zeros(nfeat, numel(cfg.lambda));
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
            cfg.weights = all_weights(CV.training(ff));
             
            % Sum of instances needed for the gradient
            sumyxN = sum(YX)'/N;
            
            % --- Loop through lambdas ---
            for ll=1:numel(cfg.lambda)
                lambda = cfg.lambda(ll);
>>>>>>> preprocess
                
                % Warm-starting the initial w: wstart
                if ll==1
                    wstart = w0;
<<<<<<< HEAD
                elseif param.predict_regularisation_path ...
                        && ll>param.polyorder+1      % make sure that enough terms have been calculated
                    % Fit polynomial to regularisation path
                    % and predict next w(lambda_k)
                    quad = qpred(ll-param.polyorder-1:ll-1,:)\(ws(:,ll-param.polyorder-1:ll-1)');
=======
                elseif cfg.predict_regularisation_path ...
                        && ll>cfg.polyorder+1      % make sure that enough terms have been calculated
                    % Fit polynomial to regularisation path
                    % and predict next w(lambda_k)
                    quad = qpred(ll-cfg.polyorder-1:ll-1,:)\(ws(:,ll-cfg.polyorder-1:ll-1)');
>>>>>>> preprocess
                    wstart = ( repmat(log(lambda),[1,numel(polyvec)]).^polyvec * quad)';
                else
                    % Use the result obtained in the previous step lambda_k-1
                    wstart = ws(:,ll-1);
                end
<<<<<<< HEAD
                if param.plot
                    wspred(:,ll)= wstart;
                    [ws(:,ll),iter_tmp(ll),delta(ll)] = TrustRegionDoglegGN(logfun, wstart, param.tolerance, param.max_iter,ll);
                else
                    ws(:,ll) = TrustRegionDoglegGN(logfun, wstart, param.tolerance, param.max_iter,ll);
                end
            end
            if param.plot
=======
                if cfg.plot
                    wspred(:,ll)= wstart;
                    [ws(:,ll),iter_tmp(ll),delta(ll)] = TrustRegionDoglegGN(logfun, wstart, cfg.tolerance, cfg.max_iter,ll);
                else
                    ws(:,ll) = TrustRegionDoglegGN(logfun, wstart, cfg.tolerance, cfg.max_iter,ll);
                end
            end
            if cfg.plot
>>>>>>> preprocess
                delta = delta + delta_tmp;
                iter = iter + iter_tmp;
                C = C + corr(ws);
            end
            
            cl = clabel(CV.test(ff));
            % Calculate classification accuracy by multiplying decision values
            % with the class label
<<<<<<< HEAD
            acc = acc + sum( (X0(CV.test(ff),:) * ws) .* repmat(cl(:),[1,numel(param.lambda)]) > 0)' / CV.TestSize(ff);
        end
        
        acc = acc / param.k;
=======
            acc = acc + sum( (X0(CV.test(ff),:) * ws) .* repmat(cl(:),[1,numel(cfg.lambda)]) > 0)' / CV.TestSize(ff);
        end
        
        acc = acc / cfg.k;
>>>>>>> preprocess
        
        [~, best_idx] = max(acc);
        
        % Diagnostic plots if requested
<<<<<<< HEAD
        if param.plot
=======
        if cfg.plot
>>>>>>> preprocess
            figure,
            nCol=3; nRow=1;
            subplot(nRow,nCol,1),imagesc(C); title({'Mean correlation' 'between w''s'}),xlabel('lambda#')
            subplot(nRow,nCol,2),plot(delta),title({'Mean trust region' 'size at termination'}),xlabel('lambda#')
<<<<<<< HEAD
            subplot(nRow,nCol,3),plot(iter/param.k),hold all,
=======
            subplot(nRow,nCol,3),plot(iter/cfg.k),hold all,
>>>>>>> preprocess
            title({'Mean number' 'of iterations (across folds)'}),xlabel('lambda#')
            
            % Plot regularisation path (for the last training fold)
            figure
<<<<<<< HEAD
            for ii=1:nfeat, semilogx(param.lambda,ws(ii,:),'o-','MarkerFaceColor','w'), hold all, end
=======
            for ii=1:nfeat, semilogx(cfg.lambda,ws(ii,:),'o-','MarkerFaceColor','w'), hold all, end
>>>>>>> preprocess
            plot(xlim,[0,0],'k-'),title('Regularisation path for last iteration'),xlabel('lambda#')
            
            % Plot cross-validated classification performance
            figure
<<<<<<< HEAD
            semilogx(param.lambda,acc)
            title([num2str(param.k) '-fold cross-validation performance'])
            hold all
            plot([param.lambda(best_idx), param.lambda(best_idx)],ylim,'r--'),plot(param.lambda(best_idx), acc(best_idx),'ro')
=======
            semilogx(cfg.lambda,acc)
            title([num2str(cfg.k) '-fold cross-validation performance'])
            hold all
            plot([cfg.lambda(best_idx), cfg.lambda(best_idx)],ylim,'r--'),plot(cfg.lambda(best_idx), acc(best_idx),'ro')
>>>>>>> preprocess
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
    
<<<<<<< HEAD
    lambda = param.lambda(best_idx);
    param.weights = all_weights;    
=======
    lambda = cfg.lambda(best_idx);
    cfg.weights = all_weights;    
>>>>>>> preprocess
end

%% Train classifier on the full training data
YX = Y*X0;
<<<<<<< HEAD
sumyxN = sum(param.weights .* YX)'/N;
X = X0;

w = TrustRegionDoglegGN(logfun, w0, param.tolerance, param.max_iter, 1);
=======
sumyxN = sum(cfg.weights .* YX)'/N;
X = X0;

w = TrustRegionDoglegGN(logfun, w0, cfg.tolerance, cfg.max_iter, 1);
>>>>>>> preprocess

%% Set up classifier struct
cf = struct();

<<<<<<< HEAD
if param.bias > 0
=======
if cfg.bias > 0
>>>>>>> preprocess
    cf.w = w(1:end-1);
    cf.b = w(end);
    
    % Bias term needs correct scaling 
<<<<<<< HEAD
    cf.b = cf.b * param.bias;
=======
    cf.b = cf.b * cfg.bias;
>>>>>>> preprocess
else
    cf.w = w;
    cf.b = 0;
end

<<<<<<< HEAD
if strcmp(param.reg,'l2')
=======
if strcmp(cfg.reg,'l2')
>>>>>>> preprocess
    cf.lambda = lambda;
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

%     function [f,g,h] = lr_objective(w)
<<<<<<< HEAD
%         % Evaluate exponential for all samples
=======
%         % Evaluate exponential for all instances
>>>>>>> preprocess
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
<<<<<<< HEAD
%         % Evaluate exponential for all samples
=======
%         % Evaluate exponential for all instances
>>>>>>> preprocess
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
<<<<<<< HEAD
        g = ((param.weights .* sigma)' * YX)'/N - sumyxN;
        
        % Hessian
        if nargout>1
            h = bsxfun(@times, X, (param.weights .* sigma) .* (1 - sigma))' * X/N;  % faster to first multiply X by sigma(1-sigma)
=======
        g = ((cfg.weights .* sigma)' * YX)'/N - sumyxN;
        
        % Hessian
        if nargout>1
            h = bsxfun(@times, X, (cfg.weights .* sigma) .* (1 - sigma))' * X/N;  % faster to first multiply X by sigma(1-sigma)
>>>>>>> preprocess
        end
    end

    function [g,h] = lr_gradient_and_hessian_tanh_L2(w)
        % Logistic gradient and Hessian expressed using the hyperbolic 
        % tangent *with L2 regularisation term included*
        % Using the identity: 1 / (1 + exp(-x)) = 0.5 + 0.5 * tanh(x/2)
        sigma = 0.5 + 0.5 * tanh(YX*w/2);
        
        % Gradient
<<<<<<< HEAD
        g = ((param.weights .* sigma)' * YX)'/N - sumyxN + lambda * w;
        
        % Hessian
        if nargout>1
            h = lambda * I + bsxfun(@times, X, (param.weights .* sigma) .* (1 - sigma))' * X/N;  % faster to first multiply X by sigma(1-sigma)
=======
        g = ((cfg.weights .* sigma)' * YX)'/N - sumyxN + lambda * w;
        
        % Hessian
        if nargout>1
            h = lambda * I + bsxfun(@times, X, (cfg.weights .* sigma) .* (1 - sigma))' * X/N;  % faster to first multiply X by sigma(1-sigma)
>>>>>>> preprocess
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

%     function xo = logreg_fix(x)
%         % This is a fix to the logistic loss function found on
%         % http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
%         % and used in the LIBLINEAR code. It prevents exp from overflowing.
%         % Logistic loss: 1./1+exp(-YX*w);
%         xo = x;
%         xo(x>=0) = 1./(1+exp(-x(x>=0)));
%         xo(x<0)  = exp(x(x<0))./(1+exp(x(x<0)));
%     end
end