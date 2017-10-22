function cf = train_svm(cfg,X,clabel)
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
%                  Alternatively, a custom kernel can be provided if there
%                  is a function called *_kernel is in the MATLAB path, 
%                  where "*" is the name of the kernel (e.g. rbf_kernel).
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

% Vector of 1's we need for optimisation
ONE = ones(N,1);

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

% Augment X with intercept
if cfg.intercept
    X = cat(2,X, ones(N,1));
    nFeat = nFeat + 1;
end

%% Compute kernel matrix
% [shouldn't be a problem with neuroscience data since nr of samples is
% typically not too large]
Q = compute_kernel_matrix(cfg, X);

%% Absorb class labels 1 and -1 in the kernel matrix
Q = Q .* (clabel * clabel');

%% Automatically set the search grid for C
if ischar(cfg.C) && strcmp(cfg.C,'auto')
    cfg.C = logspace(-4,4,10);
end


%% Find best lambda using cross-validation
if numel(cfg.C)>1
    
    % The regularisation path for logistic regression is needed. ...
    CV = cvpartition(N,'KFold',cfg.K);
    ws = zeros(nFeat, numel(cfg.C));
    acc = zeros(numel(cfg.C),1);
    
    if cfg.plot
        C = zeros(numel(cfg.C));
    end

    % --- Start cross-validation ---
    for ff=1:cfg.K
        
        % Random order of the samples
        o = zeros(CV.TrainSize(ff)*cfg.n_epochs,1);
        for ee=1:cfg.n_epochs
            o( (ee-1)*CV.TrainSize(ff)+1:ee*CV.TrainSize(ff)) = randperm(CV.TrainSize(ff));
        end

        % --- Loop through lambdas ---
        for ll=1:numel(cfg.C)
            ws(:,ll) = optim_fun(X(CV.training(ff),:), clabel(CV.training(ff)), cfg.C(ll), o);
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
        for ii=1:nFeat, semilogx(cfg.C,ws(ii,:),'-'), hold all, end
        plot(xlim,[0,0],'k-'),title('Regularisation path for last iteration'),xlabel('lambda#')
        
        % Plot cross-validated classification performance
        figure
        semilogx(cfg.C,acc)
        title([num2str(cfg.K) '-fold cross-validation performance'])
        hold all
        plot([cfg.C(best_idx), cfg.C(best_idx)],ylim,'r--'),plot(cfg.C(best_idx), acc(best_idx),'ro')
        xlabel('Lambda'),ylabel('Accuracy'),grid on
        
        
    end
else
    % there is just one lambda: no grid search
    best_idx = 1;
end

%% Train classifier on the full training data (using the best lambda)
C = cfg.C(best_idx);

% Set up classifier
cf= [];
cf.kernel = cfg.kernel;

% Initialise alpha
alpha = zeros(N,1);

% Optimise alpha
[cf.alpha,iter] = DualCoordinateDescentL1(alpha, Q, C, ONE, cfg.tolerance);

% Save support vectors
cf.support_vector_indices = find(cf.alpha>0);
cf.support_vectors = X(cf.support_vector_indices,:);
% 
% if cfg.intercept
%     cf.alpha = alpha(1:end-1);
%     cf.b = alpha(end);
% else
%     cf.alpha = alpha;
%     cf.b = 0;
% end



end

