function cf = train_libsvm(X,clabel,param)
% Trains a support vector machine.
% Usage:
% cfy = train_libsvm(X,clabel,param)
% 
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing 
%                  1's (class 1) and -1's (class 2)
%
% param          - optional struct with hyperparameters passed on to the svmtrain
%                  function
%
% %libsvm_options:
% .s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC		(multi-class classification)
% 	1 -- nu-SVC		(multi-class classification)
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR	(regression)
% 	4 -- nu-SVR		(regression)
% .t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 	4 -- precomputed kernel (kernel values in training_instance_matrix)
% .d degree : set degree in kernel function (default 3)
% .g gamma : set gamma in kernel function (default 1/num_features)
% .r coef0 : set coef0 in kernel function (default 0)
% .c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% .n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% .p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% .m cachesize : set cache memory size in MB (default 100)
% .e epsilon : set tolerance of termination criterion (default 0.001)
% .h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% .wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
% .v n : n-fold cross validation mode
% .q : quiet mode (no outputs)
%
%Output:
% cfy - struct specifying the classifier with the following fields:
% classifier   - 'lda', type of the classifier
% svmstruct    - matlab struct with details about the trained classifier
%

% convert params struct to LIBSVM style name-value pairs
par = sprintf('-s %d -t %d -d %d -g %d -r %d -c %d -n %d -p %d -m %d -e %d -h %d -b %d -wi %d -v %d', ...
    12);
if param,'quiet')
    par= [par '-q' ];
end

cf= struct();

% Call LIBSVM training function
cf.model = svmtrain(double(clabel(:)), double(X),par);
