function perf = mv_classifier_performance(metric, cf_output, label, dim)
%Calculates a classifier performance metric such as classification accuracy
%based on the classifier output (labels or decision values).
%
%Usage:
%  perf = mv_classifier_performance(metric, cf_output, label, dim)
%
%Parameters:
% metric            - desired performance metric: 
%                     'acc': classification accuracy, i.e. the fraction
%                     correctly predicted labels labels
%                     'dval': decision values. Average dvals are calculated
%                     for each class separately. The first dimension of
%                     the output refers to the class, ie perf(1,...)
%                     refers to class +1 and perf(2,...) refers to class -1
%                     'auc': area under the ROC curve
%                     'roc': ROC curve TODO
% cf_output         - classifier output (labels or dvals)
% label             - true class labels
% dim               - index of dimension across which values are averaged
%                     (e.g. dim=2 if the second dimension is the number of
%                     repeats of a cross-validation). Default: [] (no
%                     averaging)
%
%Returns:
% perf - performance metric

if nargin<4
    dim=[];
end

% Labels should be the first dimension
if size(cf_output,1) ~= numel(label)
    error('Labels need to be the first dimension of the classifier output. You can use permute() to change the dimensions')
end

% Check whether the classifier output is given as predicted labels or 
% dvals. In the former case, it should consist of -1's and 1's only.
isLabel = all(ismember( unique(cf_output(~isnan(cf_output))), [-1 1] ));

% For some metrics dvals are required
if isLabel && any(strcmp(metric,{'dval' 'roc' 'auc'}))
    error('To calculate %s, classifier output must be given as dvals not as labels', metric)
end

% Classifier output can contains nan's, we need to remember their position
nanidx = isnan(cf_output);

% Calculate the metric
switch(metric)
    case 'acc'
        
        if isLabel
            % Compare predicted labels to the true labels
            perf = double(bsxfun(@eq, cf_output, label(:)));
        else
            % We first need to transform the classifier output into labels.
            % To this end, we multiply the the dvals by the true labels -
            % for correct classification the product is positive
            perf = double(bsxfun(@times, cf_output, label(:)) > 0);
        end
        
        % Recover the NaN's
        perf(nanidx)=nan;
        
        % Aggregate across samples
        perf = nanmean(perf,1);
        
    case 'dval'
        % Aggregate across samples, for each class separately 
        perf = cat(1,nanmean(cf_output(label==1,:,:,:,:,:),1),nanmean(cf_output(label==-1,:,:,:,:,:),1));
        
    case 'auc'
        % AUC can be calculated by sorting the dvals, traversing the
        % positive examples (class +1) and counting the number of negative
        % examples (class -1) with lower values
        [cf_output,soidx] = sort(cf_output,'descend');
        
        sz= size(cf_output);
        perf= zeros([1, sz(2:end)]);
        
        % Calculate AUC for each column of cf_output
        for ii=1:prod(sz(2:end))
            clabel = label(soidx(:,ii));
            
            % Find all class indices that do not correspond to NaNs
            isClass1Idx = find(clabel(:)== 1 & ~nanidx(soidx(:,ii),ii));
            isClass2 = (clabel(:)==-1 & ~nanidx(soidx(:,ii),ii));
            
            % Count number of False Positives with lower value
            for ix=1:numel(isClass1Idx)
                perf(1,ii) = perf(1,ii) + sum(isClass2(isClass1Idx(ix)+1:end));
            end
            
            % Correct by number of True Positives * False Positives
            perf(1,ii) = perf(1,ii)/ (numel(isClass1Idx) * sum(isClass2));
        end        
end

% Average across additional dimensions
for nn=1:numel(dim)
    perf = nanmean(perf, dim(nn));
end

perf = squeeze(perf);
