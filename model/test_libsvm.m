function [ypred,dval] = test_libsvm(cf,X)
% Applies a LIBSVM classifier to test data and produces class labels and decision values.
% 
% Usage:
% [labels,dval] = test_libsvm(cf,X)
% 
%Parameters:
% cf             - classifier. See train_libsvm
% X              - [samples x features] data matrix  -OR-
%                  [train samples x test samples] kernel matrix
%Output:
% clabel        - predicted class labels
% dval          - decision values

if cf.kernel_type == 4
    % kernel has been precomputed - we only pass on the kernel matrix, not
    % the data. We need to add a leading column of sample numbers
    n_te = size(X,1);
    [ypred, ~, dval] = svmpredict(zeros(n_te,1), [(1:n_te)', X], cf.model,'-q');
else
    % kernel is compute in svmtrain, pass on data
    [ypred, ~, dval] = svmpredict(zeros(size(X,1),1), X, cf.model,'-q');
end

if cf.svm_type < 3
    % CLASSIFICATION
    % clabel come as 0 and 1, need to translate back to 1 and 2
    ypred(ypred==0) = 2;
    
    % Note that dvals might be sign-reversed in some cases,
    % see http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f430
    % To fix this behavior, we inspect cf.Labels: Label(1) denotes the positive
    % class (should be 1)
    if cf.model.Label(1) ~= 1
        % 1 is negative class, hence we need to flip dvals
        dval = -dval;
    end
end

