function cf = train_libsvm(cfg,X,clabel)
% Trains a kernel support vector machine using LIBSVM. For installation
% details and further information see
% https://github.com/cjlin1/libsvm and
% https://www.csie.ntu.edu.tw/~cjlin/libsvm
%
% Usage:
% cfy = train_libsvm(cfg,X,clabel)
%
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% cfg          - struct with hyperparameters passed on to the svmtrain
%                  function
%
% %libsvm_options:
% .svm_type : set type of SVM (default 0)
% 	0 -- C-SVC		(multi-class classification)
% 	1 -- nu-SVC		(multi-class classification)
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR	(regression)
% 	4 -- nu-SVR		(regression)
% .kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 	4 -- precomputed kernel (kernel values in training_instance_matrix)
% .degree : set degree in kernel function (default 3)
% .gamma : set gamma in kernel function (default 1/num_features)
% .coef0 : set coef0 in kernel function (default 0)
% .cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% .nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% .epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% .cachesize : set cache memory size in MB (default 100)
% .eps : set tolerance of termination criterion (default 0.001)
% .shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
% .probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% .weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
% .cv : n-fold cross validation mode
% .q : quiet mode (no outputs) => for quiet mode, set .quiet = 1
%
%Output:
% cf - struct specifying the classifier
%
% Reference:
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
% vector machines. ACM Transactions on Intelligent Systems and
% Technology, 2:27:1--27:27, 2011. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm
%

% convert params struct to LIBSVM style name-value pairs
libsvm_options = sprintf('-s %d -t %d -d %d -r %d -c %d -n %d -p %d -m %d -e %d -h %d -b %d -wi %d', ...
    cfg.svm_type, cfg.kernel_type, cfg.degree, cfg.coef0, cfg.cost, cfg.nu, ...
    cfg.epsilon, cfg.cachesize, cfg.eps, cfg.shrinking, cfg.probability_estimates, ...
    cfg.weight);

if ~isempty(cfg.gamma)
    libsvm_options= [libsvm_options ' -g ' num2str(cfg.gamma)];
end
if ~isempty(cfg.cv)
    libsvm_options= [libsvm_options ' -v ' num2str(cfg.cv)];
end
if cfg.quiet
    libsvm_options= [libsvm_options ' -q' ];
end

%% Perform averaging of instances in high-dimensional feature space using the kernel trick
%% (=averaging of the kernel matrix)
%% To use this option, a precomputed kernel has to be provided (kernel_type=4)
%% and l has to be set to the number of instances to average

if cfg.l > 1
    
    nclasses = max(clabel);
    [nsamples, ~] = size(X);
    
    % Number of samples per class
    npc = arrayfun(@(c) sum(clabel == c), 1:nclasses);

    % Nr of average samples per class
    nav = floor(npc / cfg.l);
    
    % Build new kernel matrix by first averaging the rows
    tmp = zeros(sum(nav), nsamples);
    
    % Define which sets of trials get averaged
    sets = cell(nclasses, 1);
    
    row_idx = 1;
    
    for cc=1:nclasses
        class_idx = find(clabel==cc);       % find instances of current class
        
        % Create averages
        choose_idx = randperm(npc(cc), nav(cc)*cfg.l);
        sets{cc} = reshape( class_idx(choose_idx), cfg.l, []);
        
        for nn=1:nav(cc)
            % Create row of new kernel matrix by averaging rows of kernel_matrix
            tmp(row_idx, :) = mean( cfg.kernel_matrix(sets{cc}(:,nn),:) );
            row_idx = row_idx + 1;
        end
    end
    
    cfg.kernel_matrix = zeros(sum(nav));
    
    % Now also average columns
    col_idx = 1;
    for cc=1:nclasses
        for nn=1:nav(cc)
            % Create row of new kernel matrix by averaging rows of kernel_matrix
            cfg.kernel_matrix(:, col_idx) = mean( tmp(:, sets{cc}(:,nn)) ,2 );
            col_idx = col_idx + 1;
        end
    end
    
    % Create labels for average vector
    clabel = arrayfun(@(c) ones(nav(c),1)*c , 1:nclasses, 'Un',0);
    clabel = cat(1, clabel{:});
end

%% Train classifier
cf= struct();

% Call LIBSVM training function
if cfg.kernel_type == 4
    % kernel has been precomputed - we only pass on the kernel matrix, not
    % the data
    cf.model = svmtrain(double(clabel(:)), [ (1:size(cfg.kernel_matrix,1))', cfg.kernel_matrix], libsvm_options);
else
    % kernel is compute in svmtrain, pass on data
    cf.model = svmtrain(double(clabel(:)), double(X), libsvm_options);
end

%% Store other quantities
cf.kernel_type      = cfg.kernel_type;

% Averaging sets
if cfg.l > 1
    cf.l            = cfg.l;
    cf.sets         = sets;
    cf.nav_train    = nav;
end