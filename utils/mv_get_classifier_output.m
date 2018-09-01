function out = mv_get_classifier_output(output_type, cf, test_fun, Xtest)
%Convenience function that fetches classifier output, i.e. either the 
%predicted class labels or decision values.
%
%Usage:
% out = mv_get_classifier_output(output, cf, test_fun, Xtest)
%
%Parameters:
% output_type       - classifier output type, either of 
%                     'clabel' (predicted class labels)
%                     'dval' (decision values)
%                     'prob' (class probabilities)
% cf                - classifier, output of a train_* function
% test_fun          - function handle to a test_* function
% Xtest             - [nSamples x nFeatures] test data
%
%Returns:
% out - classifier outputs of the desired type

switch(output_type)
    case 'clabel'
        % Obtain the predicted class labels
        out = test_fun(cf,Xtest);
               
    case 'dval'
        % Note that the test function should be able to return decision
        % values as second argument. This does not necessarily work with 
        % all classifiers
        [~, out] = test_fun(cf,Xtest);
        
    case 'prob'
        % Note that the test function should be able to return probabilities 
        % as third argument. This does not necessarily work with 
        % all classifiers
        [~, ~, out] = test_fun(cf,Xtest);
end
