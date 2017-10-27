function cf = train_svm(cfg,X,clabel)
% Trains a support vector machine (SVM). The avoid overfitting, the
% classifier weights are penalised using L2-regularisation.
%
% Note: It is recommended to demean (X = demean(X,'constant')) or z-score 
% (X = zscore(X)) the data first to speed up the optimisation.
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
% Q              - optional kernel matrix with absorbed class labels, i.e. 
%                  Q = KER .* (clabel * clabel'), where KER is the kernel
%                  matrix. If Q is provided, the .kernel parameter is
%                  ignored. (Default [])
%
% Note: The regularisation parameter lambda is reciprocally related to the 
% cost parameter C used in LIBSVM/LIBLINEAR, ie C = 1/lambda roughly.
%
% Further parameters (that usually do not need to be changed):
% bias          - if >0 augments the data with a bias term equal to the
%                 value of bias:  X <- [X; bias] and augments the weight
%                 vector with the bias variable w <- [w; b].
%                 If 0, the bias is assumed to be 0. If set to 'auto', the
%                 bias is 1 for linear kernel and 0 for non-linear
%                 kernel (default 'auto')
% K             - the number of folds in the K-fold cross-validation for
%                 the lambda search (default 5)
% plot          - if a lambda search is performed, produces diagnostic
%                 plots including the regularisation path and
%                 cross-validated accuracy as a function of lambda (default
%                 0)
%
%Output:
% cf - struct specifying the classifier with the following fields:
% w            - normal to the hyperplane (for linear SVM)
% b            - bias term, setting the threshold
% alpha        - dual vector specifying the weighting of training samples
%                for evaluating the classifier. For alpha > 0 a particular
%                sample is a support vector
%
% IMPLEMENTATION DETAILS:
% A Dual Coordinate Descent algorithm is used to find the optimal alpha
% (weights on the samples).  *** TODO ***


% (c) Matthias Treder 2017

[N, nFeat] = size(X);

% Vector of 1's we need for optimisation
ONE = ones(N,1);

% Make sure labels come as column vector
clabel = double(clabel(:));

% Need class labels 1 and -1 here
clabel(clabel == 2) = -1;

%% Bias
if ischar(cfg.bias) && strcmp(cfg.bias,'auto')
    if strcmp(cfg.kernel,'linear')
        cfg.bias = 1;
    else
        cfg.bias = 0;
    end
end

% Augment X with bias
if cfg.bias > 0
    X = cat(2,X, ones(N,1) * cfg.bias );
    nFeat = nFeat + 1;
end


%% Precompute and regularise kernel

if isempty(cfg.Q)
    
    % Kernel function
    kernelfun = eval(['@' cfg.kernel '_kernel']);
    
    % Compute kernel matrix
    Q = kernelfun(cfg, X);
    
    % Regularise
    if cfg.regularise_kernel > 0
        Q = Q + cfg.regularise_kernel * eye(size(X,1));
    end
    
    % Absorb class labels 1 and -1 in the kernel matrix
    Q = Q .* (clabel * clabel');
    
else
    Q = cfg.Q;
end

%% Automatically set the search grid for C
if ischar(cfg.C) && strcmp(cfg.C,'auto')
    cfg.C = logspace(-4,4,10);
end

%% Find best lambda using cross-validation
if numel(cfg.C)>1
    
    %%% --- TODO ---
    
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

% Solve the dual problem and obtain alpha
alpha = DualCoordinateDescentL1(Q, C, ONE, cfg.tolerance);

%% Set up classifier
cf= [];
cf.kernel = cfg.kernel;
cf.alpha  = alpha;
cf.gamma  = cfg.gamma;

if strcmp(cfg.kernel,'linear')
    % Calculate linear weights w and bias b from alpha
    cf.w = X' * (cf.alpha .* clabel(:));
    
    if cfg.bias > 0
        cf.b = cf.w(end);
        cf.w = cf.w(1:end-1);
        
        % Bias term needs correct scaling 
        cf.b = cf.b * cfg.bias;
    else
        cf.b = 0;
    end
else
    % Nonlinear kernel: also need to save support vectors
    cf.support_vector_indices = find(cf.alpha>0);
    cf.support_vectors  = X(cf.support_vector_indices,:);
    
    % Class labels for the support vectors
    cf.y                = clabel(cf.support_vector_indices);
    
    % For convenience we save the product alpha * y for the support vectors
    cf.alpha_y = cf.alpha(cf.support_vector_indices) .* cf.y(:);
    
    if cfg.bias > 0
        %%% ???
        cf.b = 0;
        
        % remove bias part from support vectors
        cf.support_vectors = cf.support_vectors(:,1:end-1);
    else
        cf.b = 0;
    end
    
    % Save kernel function
    cf.kernelfun = kernelfun;
    
end

end