function [predlabel,dval,prob] = test_svm(cf,X)
% Applies a SVM to test data and produces class labels and decision values.
% 
% Usage:
% [predlabel,dval] = test_svm(cf,X)
% 
%Parameters:
% cf             - classifier. See train_svm
% X              - [samples x features] matrix of test data
%
%Output:
% predlabel     - predicted class labels (1's and 2's)
% dval          - decision values, i.e. distances to the hyperplane. If
%                 cf.prob==1, dval contains probabilities. Note that
%                 predicted class labels are always based on the distances
%                 to the hyperplane.
% prob          - class probabilities

if strcmp(cf.kernel,'linear')
    dval = X*cf.w + cf.b;
else
    if strcmp(cf.kernel,'precomputed')
        dval = X(:,cf.support_vector_indices) * cf.alpha_y   + cf.b;
    else
        if cf.bias > 0
            dval = cf.kernelfun(cf, cat(2,X, ones(size(X,1),1) * cf.bias ), cf.support_vectors) * cf.alpha_y   + cf.b;
        else
            dval = cf.kernelfun(cf, X, cf.support_vectors) * cf.alpha_y   + cf.b;
        end
    end
end

if cf.prob == 1
   % Calculate probabilities using Platt's approximation
   prob = 1 ./ (1 + exp( dval * cf.A + cf.B ));
end

predlabel= double(dval >= 0) + 2*double(dval < 0);