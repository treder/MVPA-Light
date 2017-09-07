function perf = mv_classifier_performance(metric, cf_output, label, dim)
%Calculates a classifier performance metric such as classification accuracy
%based on the classifier output (labels or decision values). In
%cross-validation, the metric needs to be calculated on each test fold
%separately and then averaged across folds, since a different classifier
%has been trained in each fold.
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
%                     refers to class 1 and perf(2,...) refers to class 2
%                     'auc': area under the ROC curve
%                     'roc': ROC curve TODO
% cf_output         - vector of classifier outputs (labels or dvals). If
%                     multiple test sets have been validated using
%                     cross-validation, a cell array should be provided
%                     with each cell corresponding to one test set.
% label             - vector of true class labels. If multiple test sets
%                     have been validated using cross-validation, a cell
%                     array of labels should be provided
% dim               - index of dimension across which values are averaged
%                     (e.g. dim=2 if the second dimension is the number of
%                     repeats of a cross-validation). Default: [] (no
%                     averaging)
%
%Note: cf_output is typically a cell array. The following functions provide
%classifier output as multi-dimensional cell arrays:
% - mv_crossvalidate: 2D [repeats x K] cell array
% - mv_classify_across_time: 3D [repeats x K x time points] cell array
% - mv_classify_timextime: 3D [repeat x K x time points] cell array
%
%In all three cases, however, the label array is just [repeats x K] since
%the labels are repeated for all time points. If cf_output has more
%dimensions than label, the label array is assumed to be identical across
%the extra dimensions.
%
%Returns:
% perf - performance metric

if nargin<4
    dim=[];
end

if ~iscell(cf_output), cf_output={cf_output}; end
if ~iscell(label), label={label}; end

% Check the size of the cf_output and label. nExtra keeps the number of
% elements in the extra dimensions if ndims(cf_output) > ndims(label).
% dimSkipToken then helps us looping across the extra dimensions
sz_cf_output = size(cf_output);
nExtra = prod(sz_cf_output(ndims(label)+1:end));
dimSkipToken = repmat({':'},[1, ndims(label)]);

% Check whether the classifier output is given as predicted labels or
% dvals. In the former case, it should consist of 1's and 2's only.
isLabel = all(ismember( unique(cf_output{1}), [1 2] ));

% For some metrics dvals are required
if isLabel && any(strcmp(metric,{'dval' 'roc' 'auc'}))
    error('To calculate %s, classifier output must be given as dvals not as labels', metric)
end

perf = nan(sz_cf_output);


% Calculate the performance metric
switch(metric)

    %%% acc: classification accuracy -------------------------------
    case 'acc'

        if isLabel
            % Compare predicted labels to the true labels. To this end, we
            % create a function that compares the predicted labels to the
            % true labels and takes the mean of the comparison. This gives
            % us the classification performance for each test fold.
            fun = @(cfo,lab) mean(bsxfun(@eq,cfo,lab(:)));
        else
            % We want class 1 labels to be positive, and class 2 labels to
            % be negative, because then their sign corresponds to the sign 
            % of the dvals. To this end, we transform the labels as 
            % label =  -label + 1.5 => class 1 will be +0.5 now, class 2
            % will be -0.5
            % We first need to transform the classifier output into labels.
            % To this end, we create a function that multiplies the the
            % dvals by the true labels - for correct classification the
            % product is positive, so compare whether the result is > 0.
            % Taking the mean of this comparison gives classification
            % performance.
            fun = @(cfo,lab) mean(bsxfun(@times,cfo,-lab(:)+1.5) > 0);
        end

        for xx=1:nExtra % Looping across the extra dimensions if cf_output is multi-dimensional
            % We use cellfun to apply the function defined above to each
            % cell. The result is then converted from a cell array to a
            % matrix
            perf(dimSkipToken{:},xx) = cell2mat(cellfun(fun, cf_output(dimSkipToken{:},xx), label, 'Un', 0));
        end

    %%% dval: average decision value for each class -------------------------------
    case 'dval'
        % Aggregate across samples, for each class separately
        perf = cat(1,nanmean(cf_output(label==1,:,:,:,:,:),1),nanmean(cf_output(label==2,:,:,:,:,:),1));

    case 'auc'
        % AUC can be calculated by sorting the dvals, traversing the
        % positive examples (class 1) and counting the number of negative
        % examples (class 2) with lower values
        [cf_output,soidx] = sort(cf_output,'descend');

        sz= size(cf_output);
        perf= zeros([1, sz(2:end)]);

        % Calculate AUC for each column of cf_output
        for ii=1:prod(sz(2:end))
            clabel = label(soidx(:,ii));

            % Find all class indices that do not correspond to NaNs
            isClass1Idx = find(clabel(:)== 1 & ~nanidx(soidx(:,ii),ii));
            isClass2 = (clabel(:)==2 & ~nanidx(soidx(:,ii),ii));

            % Count number of False Positives with lower value
            for ix=1:numel(isClass1Idx)
                perf(1,ii) = perf(1,ii) + sum(isClass2(isClass1Idx(ix)+1:end));
            end

            % Correct by number of True Positives * False Positives
            perf(1,ii) = perf(1,ii)/ (numel(isClass1Idx) * sum(isClass2));
        end
end

% Average across requested dimensions
for nn=1:numel(dim)
    perf = nanmean(perf, dim(nn));
end

perf = squeeze(perf);
