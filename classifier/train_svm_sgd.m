function cf = train_svm_sgd(cfg,X,clabel)
% Trains a support vector machine (SVM). The avoid overfitting, the
% classifier weights are penalised using L2-regularisation.
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
% intercept      - augments the data with an intercept term (recommended)
%                  (default 1). If 0, the intercept is assumed to be 0
% lambda         - regularisation hyperparameter controlling the magnitude
%                  of regularisation. If a single value is given, it is
%                  used for regularisation. If a vector of values is given,
%                  5-fold cross-validation is used to test all the values
%                  in the vector and the best one is selected.
%                  If set to 'auto', a default search grid is used to
%                  automatically determine the best lambda (default 'auto')
% kernel         - kernel function:
%                  'linear'     - linear kernel, trains a linear SVM
%                  'rbf'        - radial basis function or Gaussian kernel
%                  'polynomial' - polynomial kernel
%
% Note: The regularisation parameter lambda is reciprocally related to the 
% cost parameter C used in LIBSVM/LIBLINEAR, ie C = 1/lambda roughly.
%
% Optimisation and debugging parameters (usually do not need to be changed)
% n_epochs      - number of times the full set of training samples is
%                 re-iterated. For small datasets (e.g. <50 samples),
%                 multiple iterations are necessary, for larger datasets,
%                 less iterations are necessary. If set to 'auto', the
%                 number of epochs is determined heuristically based on the
%                 size of the dataset (default 'auto')
% K             - the number of folds in the K-fold cross-validation for
%                 the lambda search
% plot          - if a lambda search is performed, produces diagnostic
%                 plots including the regularisation path and
%                 cross-validated accuracy as a function of lambda

%
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - projection vector (normal to the hyperplane)
% b            - bias term, setting the threshold

% (c) Matthias Treder 2017

[N, nFeat] = size(X);

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

% Augment X with intercept
if cfg.intercept
    X = cat(2,X, ones(N,1));
    nFeat = nFeat + 1;
end

isNonlinearProblem = ~strcmp(cfg.kernel,'linear');

% To make stochastic gradient descent somewhat less stochastic
rng(42);  % 42 is THE magic number according to a maths Prof I know ...

%% Precompute kernel matrix if necessary
% [shouldn't be a problem with neuroscience data since nr of samples is
% typically small]
if isNonlinearProblem
    ker = compute_kernel_matrix(cfg, X);
end

%% Number of epochs [reiterations of the full training data] for optimisation
if ischar(cfg.n_epochs) && strcmp(cfg.n_epochs,'auto')
    cfg.n_epochs = min(20, ceil(1000/N));
end

%% Automatically set the search grid for lambda
if ischar(cfg.lambda) && strcmp(cfg.lambda,'auto')
%     cfg.lambda = logspace(-3,2,10);
    cfg.lambda = logspace(-3,5,10);
end

if isNonlinearProblem
    % Check for additional hyperparameters
    if ischar(cfg.gamma) && strcmp(cfg.gamma,'auto')
        cfg.gamma = logspace(-3,5,5);
    end
end

%% Find best lambda using cross-validation
if numel(cfg.lambda)>1
    
    % The regularisation path for logistic regression is needed. ...
    CV = cvpartition(N,'KFold',cfg.K);
    ws = zeros(nFeat, numel(cfg.lambda));
    acc = zeros(numel(cfg.lambda),1);
    
    if cfg.plot
        C = zeros(numel(cfg.lambda));
    end

    % --- Start cross-validation ---
    for ff=1:cfg.K
        
        % Random order of the samples
        o = zeros(CV.TrainSize(ff)*cfg.n_epochs,1);
        for ee=1:cfg.n_epochs
            o( (ee-1)*CV.TrainSize(ff)+1:ee*CV.TrainSize(ff)) = randperm(CV.TrainSize(ff));
        end

        % --- Loop through lambdas ---
        for ll=1:numel(cfg.lambda)
            ws(:,ll) = optim_fun(X(CV.training(ff),:), clabel(CV.training(ff)), cfg.lambda(ll), o);
        end
        
        if cfg.plot
            C = C + corr(ws);
        end
        
        cl = clabel(CV.test(ff));
        % Calculate classification accuracy by multiplying decision values
        % with the class label
        acc = acc + sum( (X(CV.test(ff),:) * ws) .* cl(:) > 0)' / CV.TestSize(ff);
    end
    
    acc = acc / cfg.K;
    
    [~, best_idx] = max(acc);
    
    % Diagnostic plots if requested
    if cfg.plot
        figure,
        imagesc(C); title({'Mean correlation' 'between w''s'}),xlabel('lambda#')
        
        % Plot regularisation path (for the last training fold)
        figure
        for ii=1:nFeat, semilogx(cfg.lambda,ws(ii,:),'-'), hold all, end
        plot(xlim,[0,0],'k-'),title('Regularisation path for last iteration'),xlabel('lambda#')
        
        % Plot cross-validated classification performance
        figure
        semilogx(cfg.lambda,acc)
        title([num2str(cfg.K) '-fold cross-validation performance'])
        hold all
        plot([cfg.lambda(best_idx), cfg.lambda(best_idx)],ylim,'r--'),plot(cfg.lambda(best_idx), acc(best_idx),'ro')
        xlabel('Lambda'),ylabel('Accuracy'),grid on
        
        
    end
else
    % there is just one lambda: no grid search
    best_idx = 1;
end

%% Random order of the samples
o = zeros(N*cfg.n_epochs,1);
% o = randi(N, [numel(o),1]);
for ee=1:cfg.n_epochs
    o( (ee-1)*N+1:ee*N) = randperm(N);
end

%% Train classifier on the full training data (using the best lambda)
lambda = cfg.lambda(best_idx);

if isNonlinearProblem
    w = NonLinearStochasticGradientDescent(ker, clabel, lambda, o);
else
    w = StochasticGradientDescent(X, clabel, lambda, o);
end

%% Set up classifier
cf= [];
cf.kernel = cfg.kernel;

if cfg.intercept
    cf.w = w(1:end-1);
    cf.b = w(end);
else
    cf.w = w;
    cf.b = 0;
end



end

