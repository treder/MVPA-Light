function perf = mv_calculate_metric(metric, cf, test_fun, Xtest, testlabels, average)
%Calculates a desired performance metric given test data and a
%classifier.
%
%Usage:
% [X,labels] = mv_calculate_metric(method,test_fun, Xtest, testlabels)
%
%Parameters:
% metric            - desired performance metric: 'acc' (accuracy),
%                     'dval' (decision values)
% cf                - classifier, output of a train_* function
% test_fun          - classifier test function, a test_* function
% Xtest             - test data
% testlabels        - test labels according to the test data
% average           - if 1, the 
%
%Returns:
% perf - performance metric, either a vector providing a metric for each
%        test sample (average=0) or the average metric (average=1)

% Calculate the metric
switch(metric)
    case 'acc'
        % Obtain the predicted class labels
        predlabels = test_fun(cf,Xtest);
        
        % correctly predicted labels
        perf = predlabels(:)==testlabels(:);
       
    case 'dval'
        % Note that the test function should be able to return decision
        % values as second argument. This does not work with all
        % classifiers (eg Random Forests)
        [~, perf] = test_fun(cf,Xtest);
end

% Average metric across test samples
if average
    switch(metric)
        case 'acc'
            perf = mean(perf);
            
        case 'dval'
            % Decision values for class +1 should be positive, decision
            % values for class -1 should be negative, hence we multiply the
            % class labels by their respective labels first and then
            % average to get an average performance value
            perf = mean( perf(:) .* testlabels(:));
            
    end
end

