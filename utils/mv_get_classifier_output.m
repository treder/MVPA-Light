function out = mv_get_classifier_output(output, cf, test_fun, Xtest)
%Convenience function that fetches classifier output, i.e. either the 
%predicted class labels or decision values.
%
%Usage:
% out = mv_get_classifier_output(output, cf, test_fun, Xtest)
%
%Parameters:
% output            - output type, 'label' (predicted labels) or 'dval'
%                     (decision values)
% cf                - classifier, output of a train_* function
% test_fun          - classifier test function, a test_* function
% Xtest             - [nSamples x nFeatures] test data
%
%Returns:
% out - classifier outputs of the desired type

switch(output)
    case 'label'
        % Obtain the predicted class labels
        out = test_fun(cf,Xtest);
               
    case 'dval'
        % Note that the test function should be able to return decision
        % values as second argument. This does not work with all
        % classifiers (eg Random Forests)
        [~, out] = test_fun(cf,Xtest);
        
end
