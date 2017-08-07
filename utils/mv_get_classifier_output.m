function out = mv_get_classifier_output(output, cf, test_fun, Xtest)
%Convenience function that fetches classifier output, i.e. either the 
%predicted class labels or decision values. Used in
%mv_classify_across_time and mv_classify_timextime.
%
%Usage:
% out = mv_get_classifier_output(output, cf, test_fun, Xtest)
%
%Parameters:
% output            - output type, 'label' (predicted class labels) or 
%                     'dval' (decision values)
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
        % values as second argument. Not all classifiers provide decision
        % values (eg Random Forests)
        [~, out] = test_fun(cf,Xtest);       
end
