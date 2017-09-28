function cf = train_svm(cfg,X,clabel)
% Trains a linear support vector machine (SVM). The avoid overfitting, the
% classifier weights are penalised using an Euclidean penalty (L2
% regularisation). 
% It is recommended that the data is z-scored.
%
% Usage:
% cf = train_svm(cfg,X,clabel)
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
% BACKGROUND:
% Linear SVM is trained by minimising the Hinge loss function:
%
%          hinge(w,x,y) = max(0, 1 - w'x * y)
%
% where w is the classifier weights, x is the feature vector, and y is the
% class label. Including the L2 penalty, this leads to the optimisation
% problem
% 
%  w =: arg min  SUM hinge(w,x,y) + lambda * ||w||^2
%
% where the summing is across samples. This is a convex optimisation 
% problem that can be solved by unconstrained minimisation.
%
% IMPLEMENTATION DETAILS:
% The above problem is convex but the loss function is not smooth. Hence,
% gradient and Hessian cannot be determined and standard descent procedures
% do not apply. To alleviate this, Rennie and Srebro (2005) introduced a
% smooth version of the Hinge loss, called smooth Hinge:
%
%                         0             if z >= 1
% smooth_hinge(z) =    { (1-z)^2/2      if 0 < z < 1
%                         0.5 - z       if z <= 0
%
% The function replaced the linear uses a quadratic function in the
% interval [0,1] to make the function differentiable at 1. Rennie also
% introduced a more generalised smooth hinge using higher order
% polynomials to better approximate hinge loss. 
%
% However, a closer approximation can be achieved by stretching/squashing
% the quadratic function without resorting to higher-order polynomials,
% which is what I use here:
%
%                         0                    if z >= 1
% modified_hinge(z) =  { -1/(2(d-1)) (z-1)^2   if d < z < 1
%                        (1+d)/2 - z           if z <= d
%
% Here, the quadratic interpolation is limited to the [d,0] interval for
% 0<=d<1, giving a better approximation to the hinge loss.
%
% Since the loss function is now smooth, a Trust Region Dogleg algorithm 
% (TrustRegionDoglegGN.m) is used to
% optimise w. The difference of the class means serves as initial estimate
% for w. 
%
%Reference:
%Rennie and Srebro (2005). Loss Functions for Preference Levels: Regression 
%with Discrete Ordered Labels.  Proc. IJCAI Multidisciplinary Workshop on 
%Advances in Preference Handling
%
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold

% (c) Matthias Treder 2017

[N, nFeat] = size(X);
X0 = X;

lambda = cfg.lambda;
d = cfg.hinge_c;
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

%% Define some constants used in the modified hinge loss function
mod_hinge1 = - 0.5 /(cfg.hinge_c - 1);
mod_hinge2 = (1 + cfg.hinge_c) / 2; 

% absorb the class labels into X
YX = Y*X0;

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
    
    % The regularisation path for logistic regression is needed. ...
    CV = cvpartition(N,'KFold',cfg.K);
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
        % Create predictor matrix for quadratic polynomial approximation 
        % of the regularisation path
        polyvec = 0:cfg.polyorder;
        qpred = cfg.lambda(:) .^ polyvec;
    end
    
    % --- Start cross-validation ---
    for ff=1:cfg.K
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
            elseif cfg.predict_regularisation_path && ll>cfg.polyorder+1
                % Fit polynomial to regularisation path
                % and predict next w(lambda_k)
                quad = qpred(ll-cfg.polyorder-1:ll-1,:)\(ws(:,ll-cfg.polyorder-1:ll-1)');
                wstart = (lambda.^polyvec * quad)';
            else
                % Use the result obtained in the previous step lambda_k-1
                wstart = ws(:,ll-1);
            end
            if cfg.plot
                wspred(:,ll)= wstart;
                [ws(:,ll),iter_tmp(ll),delta(ll)] = TrustRegionDoglegGN(fun, wstart, cfg.tolerance, cfg.max_iter,ll);
            else
                ws(:,ll) = TrustRegionDoglegGN(fun, wstart, cfg.tolerance, cfg.max_iter,ll);
            end
        end
        if cfg.plot
            delta = delta + delta_tmp;
            iter = iter + iter_tmp;
            C = C + corr(ws);
        end
        
        cl = clabel(CV.test(ff));
        acc = acc + sum( (X0(CV.test(ff),:) * ws) .* cl(:) > 0)' / CV.TestSize(ff);
    end
    
    acc = acc / cfg.K;
    
    [~, idx] = max(acc);
    lambda = cfg.lambda(idx);
    
    % Diagnostic plots if requested
    if cfg.plot
        figure,
        nCol=3; nRow=1;
        subplot(nRow,nCol,1),imagesc(C); title({'Mean correlation' 'between w''s'}),xlabel('lambda#')
        subplot(nRow,nCol,2),plot(delta),title({'Mean trust region' 'size at termination'}),xlabel('lambda#')
        subplot(nRow,nCol,3),
        plot(iter_tmp),title({'Mean number' 'of iterations'}),xlabel('lambda#')
        
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
        
        % Plot first two dimensions
        figure
        plot(ws(1,:),ws(2,:),'ko-')
        hold all, plot(wspred(1,2:end),wspred(2,2:end),'o')
        hold on, plot(wslin(1,2:end),wslin(2,2:end),'gd')
        
    end
end

%% Train classifier on the full training data (using the optimal lambda)
YX = Y*X0;
sumyxN = sum(YX)'/N;
X = X0;

fun = @(w) lr_gradient_tanh(w);

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
%%% Linear SVM loss functions. For completeness, the original hinge loss
%%% and smooth hinge functions are provided. Only the gradient and
%%% Hessians (latter functions) are needed for optimisation.

    function f = hinge(w)
        % Loss. Note: gradient and Hessian are not defined for hinge loss 
        % (non-smooth)
        f =  max(0, 1 - w'*X);
        % Add L2 penalty
        f = f + lambda * 0.5 * (w'*w);
    end

    function f = smooth_hinge(w)
        % Smooth hinge loss according to Rennie and Srebro (2005)
        f =  zeros(1, N);
        z = YX*w;
        f( z<=0 ) = 0.5 - z(z<=0);
        f( (z>0 & z<1) ) = 0.5 * ((1 - z(z>0 & z<1)).^2);

        % Add L2 penalty
        f = f + lambda * 0.5 * (w'*w);
    end

    function f = modified_hinge(w)
        % Modification of smooth hinge with a shorter interpolation
        % interval (closer match with non-smooth hinge)
        f =  zeros(1, N);
        z = YX*w;
        f( d<=z & z<1 ) = mod_hinge1 * (z(d<=z & z<1) - 1).^2;
        f( z<=d ) = mod_hinge2 - z(z<=d);
        
        % Add L2 penalty
        f = f + lambda * 0.5 * (w'*w);
    end

    function [g,h] = modified_hinge_gradient(w)
        % Gradient and Hessian of the modified hinge
        f =  zeros(1, N);
        z = 1 - YX*w;
        f( cfg.hinge_c <= z & z<1 ) = mod_hinge1 * (z-1).^2;
        f( z <= cfg.hinge_c ) = mod_hinge2 - z;
    end


%%% Plot loss functions
% w = 1;
% YX = linspace(-1,3,N);
% lambda = 0;
% 
% figure
% plot(YX, hinge(w)), hold all
% plot(YX, smooth_hinge(w))
% plot(YX, modified_hinge(w))
% legend({'Hinge' 'Smooth hinge' 'Modified hinge'})

end

