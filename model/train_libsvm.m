function cf = train_libsvm(param,X,y)
% Trains a kernel support vector machine (SVM) or a support vector
% regression (SVR) using LIBSVM. The sym_type parameter controls whether
% classification or regression is performed, see below.
%
% For installation details and further information see
% https://github.com/cjlin1/libsvm and 
% https://www.csie.ntu.edu.tw/~cjlin/libsvm
%
% Usage:
% cf = train_libsvm(param,X,clabel)
% 
%Parameters:
% X              - [samples x features] matrix of training instances  -OR-
%                  [samples x samples] kernel matrix
% y              - [samples x 1] vector of class labels (classification)
%                                or responses (regression)
%
% param          - struct with hyperparameters passed on to LIBSVM's svmtrain
%                  function
%
% .kernel        - kernel function:
%                  'linear'     - linear kernel ker(x,y) = x' y
%                  'rbf'        - radial basis function or Gaussian kernel
%                                 ker(x,y) = exp(-gamma * |x-y|^2);
%                  'polynomial' - polynomial kernel
%                                 ker(x,y) = (gamma * x * y' + coef0)^degree
%                  'sigmoid'    - sigmoid kernel
%
%                  If a precomputed kernel matrix is provided as X, set
%                  param.kernel = 'precomputed'.
% .quiet         - if 1, the classifier is trained in quiet mode (no
%                  outputs) (default 1)
%
% further libsvm_options:
% .svm_type : set type of SVM (default 0)
% 	0 -- C-SVC		(multi-class classification)
% 	1 -- nu-SVC		(multi-class classification)
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR	(regression)
% 	4 -- nu-SVR		(regression)
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
% .cv : n-fold cross validation mode (default [] = no cross-validation)
%
%Output:
% cf - [struct] specifying the model. The result of svmtrain is stored
%      in cf.model
%
% Reference:
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
% vector machines. ACM Transactions on Intelligent Systems and
% Technology, 2:27:1--27:27, 2011. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm
%

% convert kernel parameter to the appropriate kernel type for LIBSVM
switch(param.kernel)
    case 'linear', param.kernel_type = 0;
    case 'polynomial', param.kernel_type = 1;
    case 'rbf', param.kernel_type = 2;
    case 'sigmoid', param.kernel_type = 3;
    case 'precomputed', param.kernel_type = 4;
end

% convert params struct to LIBSVM style name-value pairs
libsvm_options = sprintf('-s %d -t %d -d %d -r %d -c %d -n %d -p %d -m %d -e %d -h %d -b %d -wi %d', ...
    param.svm_type, param.kernel_type, param.degree, param.coef0, param.cost, param.nu, ...
    param.epsilon, param.cachesize, param.eps, param.shrinking, param.probability_estimates, ...
    param.weight);

if ~isempty(param.gamma)
    libsvm_options= [libsvm_options ' -g ' num2str(param.gamma)];
end
if ~isempty(param.cv)
    libsvm_options= [libsvm_options ' -v ' num2str(param.cv)];
end
if param.quiet
    libsvm_options= [libsvm_options ' -q' ];
end

% Call LIBSVM training function
cf = [];

if param.svm_type < 3
    y = double(y(:)==1);         % classification
else
    y = double(y(:));            % regression
end

if strcmp(param.kernel,'precomputed')
    % for precomputed kernels we must provide the sample number as an
    % additional column
    cf.model = svmtrain(y, [(1:size(X,1))' double(X)], libsvm_options);
    
else
    cf.model = svmtrain(y, double(X), libsvm_options);
end
% note: if svmtrain crashes for you make sure that it is not being
% overshadowed by at Matlab function of the same name ('svmtrain' was also
% the name of a Matlab function that has been replaced by 'fitcsvm' in
% later versions of Matlab)

% Save parameters needed for testing
cf.kernel           = param.kernel;
cf.kernel_type      = param.kernel_type;
cf.svm_type         = param.svm_type;

