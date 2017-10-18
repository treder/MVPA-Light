function w = StochasticGradientDescent(X, y, lambda, o)
% Implementation of a Stochastic Gradient Descent (SGD) algorithm for
% training linear SVMs, based on Shalev-Shwartz and Ben-David (2014), see
% page p.213 for the algorithm. It has been dubbed Pegasos (Primal 
% Estimated sub-gradient Solver for SVM) and it was first introduced in 
% Shalev-Shwartz, Singer and Srebro (2007).
%
% The optimisation problem is 
%
%           w = arg min  lambda/2 ||w||^2 + 1/m SUM hinge(w,xi)
%
% where 
%
%           hinge(w,xi) = max(0, 1 - yi * w'xi)
%
% is the hinge loss function.
%
% Usage: w = StochasticGradientDescent(X,lambda,n_epochs)
%
% X            - [samples x features] matrix of training samples
% y            - [samples x 1] vector of class labels (+1 and -1)
% lambda       - regularisation parameter
% o            - a vector of sample indices from the range [1,N], where N
%                is the number of samples. This is the order in which the
%                samples are drawn for training.
%
% References:
% Shalev-Shwartz and Ben-David (2014). Understanding Machine Learning: 
% From Theory to Algorithms. Cambridge University Press New York, NY, USA
%
% Shalev-Shwartz,  Singer,  &  Srebro (2007). Pegasos: Primal Estimated  
% sub-GrAdient SOlver for SVM. International  Conference  on  Machine  
% Learning, pp. 807–814.

% (c) Matthias Treder 2017

% Absorb the class labels in the data matrix X to ease the calculations
YX = diag(y) * X;

% Total number of iterations
T = length(o);

% Initialise 
% theta = 0;
w = zeros(size(X,2),1);

% Iterate
for t=1:T
    
    w = (1-1/t) * w + double(YX(o(t),:) * w < 1) * YX(o(t),:)' / (lambda*t);
    
end

% % Output average
% w = w / T;


%%% OLD VERSION
% for t=1:T
%     w = w + theta / (lambda*t);
%     
%     % Update theta if the margin is violated
%     if y(o(t)) * X(o(t),:) * w < 1
%         theta = theta + y(o(t)) * X(o(t),:)';
%     end
% end
