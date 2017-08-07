function perf = mv_metric_calculate(metric, cf, test_fun, Xtest, testlabels, average)
%Calculates a desired performance metric given test data and a
%classifier.
%
%Usage:
% [X,labels] = mv_metric_calculate(method,test_fun, Xtest, testlabels)
%
%Parameters:
% metric            - desired performance metric: 'acc' (accuracy),
%                     'dval' (decision values)
% cf                - classifier, output of a train_* function
% test_fun          - classifier test function, a test_* function
% Xtest             - [nSamples x nFeatures] test data
% testlabels        - test labels according to the test data. If number of
%                     test samples is larger than the number of test
%                     labels, it is assumed 
% average           - if 1, the performance metric is averaged across
%                     samples
%
%Returns:
% perf - performance metric, either a vector providing a metric for each
%        test sample (average=0) or the average metric (average=1)

% If the sizes of Xtest and test labels do not match it is assumed that
% test samples for different time points/frequencies etc. are provided at
% once. The predicted labels are then reshaped 
if size(Xtest,1) ~= numel(testlabels)
    nRepeats = size(Xtest,1) / numel(testlabels);
else
    nRepeats = 1;
end

% Calculate the metric
switch(metric)
    case 'acc'
        % Obtain the predicted class labels
        predlabels = test_fun(cf,Xtest);
        
        % If the test samples stem from multiple time points/frequencies,
        % we must reshape the predlabels into a matrix
        predlabels = reshape(predlabels, [], nRepeats);
        
        % correctly predicted labels
%         perf = predlabels(:)==testlabels(:);
        perf = bsxfun(@eq, predlabels, testlabels(:));
       
    case 'dval'
        % Note that the test function should be able to return decision
        % values as second argument. This does not work with all
        % classifiers (eg Random Forests)
        [~, perf] = test_fun(cf,Xtest);
        
        % If the test samples stem from multiple time points/frequencies,
        % we must reshape the predlabels into a matrix
        perf = reshape(perf, [], nRepeats);
       
end

% Average metric across test samples
if average
    switch(metric)
        case 'acc'
            perf = mean(perf,1);
            
        case 'dval'
            % Decision values for class +1 should be positive, decision
            % values for class -1 should be negative, hence we multiply the
            % class labels by their respective labels first and then
            % average to get an average performance value
            perf = mean( bsxfun( perf, testlabels(:)));
            
    end
end

