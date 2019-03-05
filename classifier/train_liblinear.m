function cf = train_liblinear(cfg,X,clabel)
% Trains a linear support vector machine of logistic regression using
% LIBLINEAR. For installation details and further information see
% https://github.com/cjlin1/liblinear and 
% https://www.csie.ntu.edu.tw/~cjlin/liblinear/
%
% Usage:
% cfy = train_liblinear(cfg,X,clabel)
% 
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels
%
% cfg          - struct with hyperparameters passed on to the train 
%                  function of LIBLINEAR
%
% .type : set type of solver (default 1)
%   for multi-class classification
% 	 0 -- L2-regularized logistic regression (primal)
% 	 1 -- L2-regularized L2-loss support vector classification (dual)
% 	 2 -- L2-regularized L2-loss support vector classification (primal)
% 	 3 -- L2-regularized L1-loss support vector classification (dual)
% 	 4 -- support vector classification by Crammer and Singer
% 	 5 -- L1-regularized L2-loss support vector classification
% 	 6 -- L1-regularized logistic regression
% 	 7 -- L2-regularized logistic regression (dual)
%   for regression
% 	11 -- L2-regularized L2-loss support vector regression (primal)
% 	12 -- L2-regularized L2-loss support vector regression (dual)
% 	13 -- L2-regularized L1-loss support vector regression (dual)
% .cost : set the parameter C (default 1)
% .epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% .eps: set tolerance of termination criterion
% 	-s 0 and 2
% 		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
% 		where f is the primal function and pos/neg are # of
% 		positive/negative data (default 0.01)
% 	-s 11
% 		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)
% 	-s 1, 3, 4 and 7
% 		Dual maximal violation <= eps; similar to libsvm (default 0.1)
% 	-s 5 and 6
% 		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
% 		where f is the primal function (default 0.01)
% 	-s 12 and 13\n"
% 		|f'(alpha)|_1 <= eps |f'(alpha0)|,
% 		where f is the dual function (default 0.1)
% .bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
% .weight: weights adjust the parameter C of different classes (see README for details)
% .cv: n-fold cross validation mode
% .c : find parameter C (only for -s 0 and 2)
% .quiet : quiet mode (no outputs)
%
%Output:
% cf - struct specifying the classifier
%
% Reference:
% R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.
% LIBLINEAR: A Library for Large Linear Classification, Journal of
% Machine Learning Research 9(2008), 1871-1874. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/liblinear
%

if ~any(cfg.type == [0,2]) && any(cfg.c==1)
    error('Warm-start parameter search only available for type 0 and type 2')
end

% convert params struct to LIBLINEAR style name-value pairs
liblinear_options = sprintf('-s %d -p %d -B %d', ...
    cfg.type, cfg.epsilon, cfg.bias);

if ~isempty(cfg.eps)
    liblinear_options= [liblinear_options ' -e ' num2str(cfg.eps)];
end
if ~isempty(cfg.weight)
    liblinear_options= [liblinear_options ' -wi ' num2str(cfg.weight)];
end
if ~isempty(cfg.cv)
    liblinear_options= [liblinear_options ' -v ' num2str(cfg.cv)];
end
if cfg.quiet
    liblinear_options= [liblinear_options ' -q' ];
end

if ~isempty(cfg.c)
    % First run cross-validation to find best cost parameter C
    cfg.cost = train(double(clabel(:)), sparse(X), [liblinear_options ' -C']);
end

if ~isempty(cfg.cost)
    % Set cost parameter to either default or to the cross-validated
    % version
    liblinear_options= [liblinear_options ' -c ' num2str(cfg.cost(1))];
end

% Call LIBLINEAR training function
cf = train(double(clabel(:)==1), sparse(X), liblinear_options);
