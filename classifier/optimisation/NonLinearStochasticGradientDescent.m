function w = NonLinearStochasticGradientDescent(ker, y, lambda, o)
% Implementation of a non-linear Stochastic Gradient Descent (SGD) 
% algorithm for training linear SVMs, based on Shalev-Shwartz and 
% Ben-David (2014), see page p.223 for the algorithm. It has been dubbed 
% Pegasos (Primal Estimated sub-gradient Solver for SVM) and it was first 
% introduced in Shalev-Shwartz, Singer and Srebro (2007).
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
% Usage: w = NonLinearStochasticGradientDescent(X,lambda,n_epochs)
%
% ker          - [samples x samples] kernel matrix
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

% Total number of iterations
T = length(o);

% Initialise 
alpha = zeros(size(ker,1),1);
beta = alpha;

% Do not want to keep all the alpha's for all iterations to save memory, so
% we flip alpha and alpha_old
alpha_old = alpha;

% Iterate
for t=1:T
    
    alpha = alpha + beta / (lambda*t);
    
    if y(o(t)) * ker(o(t),:) * (alpha - alpha_old) < 1
        beta(o(t)) = beta(o(t)) + y(o(t));
        alpha_old = alpha;
    end
    
end

% Output average
alpha = alpha / T;

