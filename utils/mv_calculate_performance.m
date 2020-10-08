function [perf, perf_std] = mv_calculate_performance(metric, output_type, model_output, y, dim)
%Calculates a classifier performance metric such as classification accuracy
%based on the classifier output (e.g. labels or decision values). In
%cross-validation, the metric needs to be calculated on each test fold
%separately and then averaged across folds, since a different classifier
%has been trained in each fold.
%
%Usage:
%  [perf, perf_std] = mv_classifier_performance(metric, model_output, y, dim)
%
%Parameters:
% metric            - desired performance metric
%                     Classification metrics:
%                     'acc': classification accuracy, i.e. the fraction
%                     correctly predicted labels labels
%                     'dval': decision values. Average dvals are calculated
%                     for each class separately. The first dimension of
%                     the output refers to the class, ie perf(1,...)
%                     refers to class 1 and perf(2,...) refers to class 2
%                     'tval': independent samples t-test statistic for
%                     unequal sample sizes. It is calculated across the 
%                     distribution of dvals for two classes
%                     'auc': area under the ROC curve
%                     'confusion': confusion matrix (needs class labels) as
%                     classifier output.
%                     'kappa'
%                     'precision'
%                     'recall'
%                     'f1'
%                     'none'
%
%                     Regression metrics:
%                     'mse' (or 'mean_squared_error')
%                     'mae' (or 'mean_absolute_error')
% output_type       - type of classifier output ('clabel', 'dval', or 'prob'). 
%                     See mv_get_classifier_output for details. Not
%                     relevant for regression.
% model_output      - vector of classifier or regression model outputs. If
%                     multiple test sets have been validated using
%                     cross-validation, a (possibly mult-dimensional)
%                     cell array should be provided with each cell
%                     corresponding to one test set.
% y                 - vector of true class labels (classification) or true
%                     responses (regression). If multiple test sets
%                     have been validated using cross-validation, a cell
%                     array should be provided (same size as
%                     model_output)
% dim               - index of dimension across which values are averaged
%                     (e.g. dim=2 if the second dimension is the number of
%                     repeats of a cross-validation). Default: [] (no
%                     averaging).
%                     Note that a weighted mean is used, that is, folds
%                     are weighed proportionally to their number of test
%                     samples.
%
%Note: model_output is typically a cell array. The following functions provide
%classifier output as multi-dimensional cell arrays:
% - mv_crossvalidate: 2D [repeats x K] cell array
% - mv_classify_across_time: 3D [repeats x K x time points] cell array
% - mv_classify_timextime: 3D [repeat x K x time points] cell array
% - mv_classify: [repeat x K x ... x ... ] of any dimension >= 2
%
% In all three cases, however, the corresponding y array is just
% [repeats x K] since class labels are repeated for all time points. If 
% model_output has more dimensions than y, the y array is assumed 
% to be identical across the extra dimensions.
%
% Each cell should contain a [testlabel x 1] vector of classifier outputs
% (ie labels or dvals) for the according test set. Alternatively, it can
% contain a [testlabel x ... x ...] matrix, like the output of
% mv_classify_timextime. But then all other dimensions need to have the
% same size.
%
%Reference: 
% For the multi-class generalisation of the metrics the definitions
% in this paper are followed:
% Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance 
% measures for classification tasks. Information Processing & Management, 
% 45(4), 427–437. https://doi.org/10.1016/J.IPM.2009.03.002
%
%Returns:
% perf     - performance metric
% perf_std - standard deviation of the performance metric across folds 
%            and repetitions, indicating the variability of the metric.

% (c) Matthias Treder

if nargin<5
    dim=[];
end

if ~iscell(model_output), model_output={model_output}; end
if ~iscell(y), y={y}; end

% Check the size of the model_output and y. nextra keeps the number of
% elements in the extra dimensions if ndims(model_output) > ndims(y). For
% instance, if we classify across time, model_output is [repeats x folds x time]
% and y is [repeats x folds] so we have 1 extra dimension (time).
sz_output = size(model_output);
nextra = prod(sz_output(ndims(y)+1:end));
% dimSkipToken helps us looping across the extra dimensions
dim_skip_token = repmat({':'},[1, ndims(y)]);

% For some metrics dvals or probabilities are required
if strcmp(output_type,'clabel') && any(strcmp(metric,{'dval' 'roc' 'auc'}))
    error('To calculate dval/roc/auc, classifier output must be given as dvals or probabilities, not as class labels')
end

perf = cell(sz_output);
nclasses = max(max(max( vertcat(y{:}) )));

% Calculate the requested performance metric
switch(metric)
    
    %%% ------ CLASSIFICATION METRICS -----
    case {'acc', 'accuracy'}
        %%% ------ ACC: classification accuracy -----
        
        if strcmp(output_type,'clabel')
            % Compare predicted labels to the true labels. To this end, we
            % create a function that compares the predicted labels to the
            % true labels and takes the mean of the comparison. This gives
            % us the classification performance for each test fold.
            fun = @(cfo,lab) mean(bsxfun(@eq,cfo,lab(:)), 1);
        elseif strcmp(output_type,'dval')
            % We want class 1 labels to be positive, and class 2 labels to
            % be negative, because then their sign corresponds to the sign
            % of the dvals. To this end, we transform the labels as
            % y =  -y + 1.5 => class 1 will be +0.5 now, class 2
            % will be -0.5
            % We first need to transform the classifier output into labels.
            % To this end, we create a function that multiplies the the
            % dvals by the true labels - for correct classification the
            % product is positive, so compare whether the result is > 0.
            % Taking the mean of this comparison gives classification
            % performance.
            fun = @(cfo,lab) mean(bsxfun(@times, cfo, -lab(:)+1.5) > 0, 1);
        elseif strcmp(output_type,'prob')
            % Probabilities represent the posterior probability for class
            % being class 1, ranging from 0 to 1. To transform 
            % probabilities into class labels, subtract 0.5 from the
            % probabilities and also transform the labels (see previous
            % paragraph about dvals for details)
            fun = @(cfo,lab) mean(bsxfun(@times, cfo-0.5, -lab(:)+1.5) > 0, 1);
        end
        
        % Looping across the extra dimensions if model_output is multi-dimensional
        for xx=1:nextra
            % Use cellfun to apply the function defined above to each cell
            perf(dim_skip_token{:},xx) = cellfun(fun, model_output(dim_skip_token{:},xx), y, 'Un', 0);
        end
        
    case 'auc'
        %%% ----- AUC: area under the ROC curve -----
        
        % AUC can be calculated by sorting the dvals, traversing the
        % positive examples (class 1) and counting the number of negative
        % examples (class 2) with lower values
        
        % There is different ways to calculate AUC. An efficient one for
        % our purpose is to sort the y vector in an ascending fashion,
        % the dvals are later sorted accordingly using soidx.
        [y, soidx] = cellfun(@(lab) sort(lab,'ascend'), y, 'Un',0);
        N1 = cellfun( @(lab) {sum(lab==1)}, y);
        N2 = cellfun( @(lab) {sum(lab==2)}, y);
        
        % Anonymous function gtfun takes a specific row cii and checks how many
        % times the value (corresponding to class label 1) in one column is greater than the value in the
        % rows of the submatrix cn1 (corresponding to the class labels 2). Ties are weighted by 0.5. This gives a
        % count for each column separately.
        % cii and cn1 represent the possibly matrix-sized content of a cell
        % of model_output
        gtfun = @(cii, cn1) (sum(bsxfun(@gt, cii, cn1) + 0.5*bsxfun(@eq,cii,cn1) ));
        % arrsum repeats gtfun for each sample in matrix c corresponding to
        % class 1. The counts are then summed across the class 1 samples
        arrsum =  @(c,n1) sum(cell2mat(   arrayfun(@(ii) gtfun(c(ii,:,:,:,:,:,:,:),c(n1+1:end,:,:,:,:,:,:,:)), 1:n1,'Un',0)'    ) );
        for xx=1:nextra
            % Sort decision values using the indices of the sorted labels.
            % Add a bunch of :'s to make sure that we preserve the other
            % (possible) dimensions
            cf_so = cellfun(@(c,so) c(so,:,:,:,:,:,:,:,:,:) , model_output(dim_skip_token{:},xx), soidx, 'Un',0);
            % Use cellfun to perform the following operation within each
            % cell:
            % For each class 1 sample, we count how many class 2 exemplars
            % have a lower decision value than this sample. If there is a
            % tie (dvals equal for two exemplars of different classes),
            % we add 0.5. Dividing that number by the #class 1 x #class 2
            % (n1*n2) gives the AUC.
            % We do this by applying the above-defined arrsum to every cell
            % in model_output and then normalising by (n1*n2)
            perf(dim_skip_token{:},xx) = cellfun(@(c,n1,n2) arrsum(c,n1)/(n1*n2), cf_so, N1,N2, 'Un',0);
        end
        
    case 'confusion'
        %%% ---------- confusion matrix ---------
        
        if ~strcmp(output_type,'clabel')
            error('confusion matrix requires class labels as classifier output_type')
        end

        perf = calculate_confusion_matrix(1);

    case 'dval'
        %%% ------ DVAL: average decision value for each class ------
        
        perf = cell([sz_output,2]);
        
        % Aggregate across samples, for each class separately
        if nextra == 1
            perf(dim_skip_token{:},1) = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:,:,:,:,:),1), model_output, y, 'Un',0);
            perf(dim_skip_token{:},2) = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:,:,:,:,:),1), model_output, y, 'Un',0);
        else
            for xx=1:nextra % Looping across the extra dimensions if model_output is multi-dimensional
                perf(dim_skip_token{:},xx) = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:,:,:,:,:),1), model_output(dim_skip_token{:},xx), y, 'Un',0);
                perf(dim_skip_token{:},xx+nextra) = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:,:,:,:,:),1), model_output(dim_skip_token{:},xx), y, 'Un',0);
            end
        end
        
    case 'f1'
        %%% ---------- F1 score ---------
        
        % F1 is a weighted average between precision and recall, hence we
        % need to calculate them first
        conf = calculate_confusion_matrix(0);
        precision = calculate_precision(conf);
        recall = calculate_recall(conf);
        
        % Formula: F1 =  2*(Precision*Recall ) / (Precision + Recall)
        perf = cellfun( @(p, r) 2* (p.*r) ./ (p+r) , precision, recall, 'Un', 0);

    case 'kappa'
        %%% ---------- kappa ---------

        perf = cell(sz_output);

        % Cohen's kappa is based on observed and expected accuracy, which
        % can be obtained from the confusion matrix
        conf = calculate_confusion_matrix(0);
        
        % Observed accuracy equals standard accuracy
        % Expected accuracy is what any random classifier would be expected 
        % to attain. It is calculated as the sum of the row marginals x 
        % column marginal, divided by the total number squared
        % Kappa is given by (observed accuracy - expected accuracy)/(1 - expected accuracy)
        kappa = @(po, pe) (po - pe)./(1 - pe);
                
        for xx=1:nextra % Looping across the extra dimensions if model_output is multi-dimensional
            if ismatrix(conf{1,1,xx})
                observed_accuracy = cellfun(@(c) trace(c)/sum(sum(c)), conf(dim_skip_token{:},xx), 'Un',0);
                expected_accuracy = cellfun(@(c) (sum(c,1) * sum(c,2))/sum(c(:))^2, conf(dim_skip_token{:},xx), 'Un',0);
                perf(dim_skip_token{:},xx) = cellfun( @(po,pe) kappa(po,pe) , observed_accuracy, expected_accuracy,'Un',0);
            else
                % the content is a [classes x nclasses x ...]
                % multidimensional array, so we need to iterate
                % across the last dimension using a nested arrayfun
                % and a nested cell2mat to convert it back to a
                % matrix
                observed_accuracy = cellfun(@(c) cell2mat(arrayfun(@(ix) trace(c(:,:,ix))/sum(sum(c(:,:,ix))), 1:(numel(c)/nclasses/nclasses),'Un',0)), conf(dim_skip_token{:},xx), 'Un',0);
                expected_accuracy = cellfun(@(c) cell2mat(arrayfun(@(ix) (sum(c(:,:,ix),1) * sum(c(:,:,ix),2))/sum(sum(c(:,:,ix)))^2, 1:(numel(c)/nclasses/nclasses),'Un',0)), conf(dim_skip_token{:},xx), 'Un',0);
                perf(dim_skip_token{:},xx) = cellfun( @(po,pe) kappa(po,pe) , observed_accuracy, expected_accuracy,'Un',0);
            end
        end

    case 'tval'
        %%% ------ TVAL: independent samples t-test values ------
        % Using the formula for unequal sample size, equal variance: 
        % https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes.2C_equal_variance
        perf = cell(sz_output);
        
         % Aggregate across samples, for each class separately
        if nextra == 1
            % Get means
            M1 = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:,:,:,:,:),1), model_output, y, 'Un',0);
            M2 = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:,:,:,:,:),1), model_output, y, 'Un',0);
            % Variances
            V1 = cellfun( @(cfo,lab) nanvar(cfo(lab==1,:,:,:,:,:,:,:,:,:)), model_output, y, 'Un',0);
            V2 = cellfun( @(cfo,lab) nanvar(cfo(lab==2,:,:,:,:,:,:,:,:,:)), model_output, y, 'Un',0);
            % Class frequencies
            N1 = cellfun( @(lab) sum(lab==1), y, 'Un',0);
            N2 = cellfun( @(lab) sum(lab==2), y, 'Un',0);
            % Pooled standard deviation
            SP = cellfun( @(v1,v2,n1,n2) sqrt( ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)  ), V1,V2,N1,N2, 'Un',0);
            % T-value
            perf = cellfun( @(m1,m2,n1,n2,sp) (m1-m2)./(sp.*sqrt(1/n1 + 1/n2)) , M1,M2,N1,N2,SP,'Un',0);
        else
            for xx=1:nextra % Looping across the extra dimensions if model_output is multi-dimensional
                % Get means
                M1 = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:,:,:,:,:),1), model_output(dim_skip_token{:},xx), y, 'Un',0);
                M2 = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:,:,:,:,:),1), model_output(dim_skip_token{:},xx), y, 'Un',0);
                % Variances
                V1 = cellfun( @(cfo,lab) nanvar(cfo(lab==1,:,:,:,:,:,:,:,:,:)), model_output(dim_skip_token{:},xx), y, 'Un',0);
                V2 = cellfun( @(cfo,lab) nanvar(cfo(lab==2,:,:,:,:,:,:,:,:,:)), model_output(dim_skip_token{:},xx), y, 'Un',0);
                % Class frequencies
                N1 = cellfun( @(lab) sum(lab==1), y, 'Un',0);
                N2 = cellfun( @(lab) sum(lab==2), y, 'Un',0);
                % Pooled standard deviation
                SP = cellfun( @(v1,v2,n1,n2) sqrt( ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)  ), V1,V2,N1,N2, 'Un',0);
                % T-value
                perf(dim_skip_token{:},xx) = cellfun( @(m1,m2,n1,n2,sp) (m1-m2)./(sp.*sqrt(1/n1 + 1/n2)) , M1,M2,N1,N2,SP,'Un',0);
            end
        end
        
    case 'precision'
        %%% ---------- precision ---------

        perf = calculate_precision(calculate_confusion_matrix(0));

    case 'recall'
        %%% ---------- recall ---------

        perf = calculate_recall(calculate_confusion_matrix(0));

    %%% ------ REGRESSION METRICS -----
    case {'mse', 'mean_squared_error'}
        %%% ---------- mean squared error ---------
        for xx=1:nextra
            perf(dim_skip_token{:},xx) = cellfun( @(ypred,ytrue) mean( (ypred - ytrue).^2, 1 ), model_output(dim_skip_token{:},xx), y, 'Un', 0);
        end

    case {'mae', 'mean_absolute_error'}
        %%% ---------- mean absolute error ---------
        for xx=1:nextra
            perf(dim_skip_token{:},xx) = cellfun( @(ypred,ytrue) mean( abs(ypred - ytrue), 1), model_output(dim_skip_token{:},xx), y, 'Un', 0);
        end

    case {'r_squared'}
        %%% ---------- R-squared: fraction variance explained ---------
        for xx=1:nextra
            perf(dim_skip_token{:},xx) = cellfun( @(ypred,ytrue) 1 - var(ypred - ytrue)./var(ytrue) , model_output(dim_skip_token{:},xx), y, 'Un', 0);
        end
        
    otherwise, error('Unknown metric: %s', metric)
end

% Convert cell array to matrix. Since each cell can also contain a multi-
% dimensional array instead of a scalar, we need to make sure that these
% arrays are correctly appended as extra dimensions.
nd = sum(size(perf{1})>1); % Number of non-singleton dimensions
% nd = find(size(perf{1})>1,1,'last'); % Number of non-singleton dimensions
if nd>0
    % There is extra non-singleton dimensions within the cells. To cope with this, we
    % prepend the dimensions of the cell array as extra singleton
    % dimensions. Eg. for a [5 x 2] cell array, we do something like
    % perf{1}(1,1,:) = perf{1} so that the content of the cell is pushed to
    % dimensions 3 and higher
    dimSkip1 = repmat({1},[1, ndims(perf)]);
    innerCellSkip = repmat({':'},[1, nd]);
    for ii=1:numel(perf)
        tmp = [];
        tmp(dimSkip1{:},innerCellSkip{:}) = perf{ii};
        perf{ii} = tmp;
    end
end

perf = cell2mat(perf);
perf_std = [];

%% Average the performance metric across test sets

% Calculate STANDARD ERROR OF THE METRIC [use unweighted averages here] of the
% performance across all dimensions of dim. Typically dim will refer to 
% folds and repetitions. To this end, we first make sure that the
% dimensions are the first dimensions
tmp = permute(perf, [dim, setdiff(1:ndims(perf), dim)]);
% Then we reshape the array so that we have all of the dim dimensions as 
% concatenated in the first dimension. Finally, we can calculated the std
% in this dimension
sz = size(tmp);
if ndims(perf) == numel(dim)
    % there is no additional dimensions, so we can just vectorise tmp
    tmp = tmp(:);
else
    tmp = reshape(tmp, [prod(sz(dim)), sz(setdiff(1:ndims(tmp),dim)) ]);
end
perf_std = nanstd(tmp, [], 1);

% Calculate MEAN OF THE METRIC across requested dimensions
for nn=1:numel(dim)
    
    % For averaging, use a WEIGHTED mean: Since some test sets may have
    % more samples than others (since the number of data points is not
    % always integer divisible by K), folds with more test samples give
    % better estimates. They should be weighted higher proportionally to
    % the number of samples.
    % To achieve this, first multiply the statistic for each fold with the
    % number of samples in this fold. Then sum up the statistic and divide
    % by the total number of samples.
    if nn==1
        num_samples = cellfun(@numel,y);
        
        % Multiply metric in each fold with its number of samples
        perf = bsxfun(@times, perf, num_samples);
    end
    
    % Sum metric and respective number of samples
    perf = nansum(perf, dim(nn));
    num_samples = nansum(num_samples, dim(nn));
    
    % Finished - we need to normalize again by the number of samples to get
    % back to the original scale
    if nn==numel(dim)
        perf = perf ./ num_samples;
    end
end

perf = squeeze(perf);
perf_std = squeeze(perf_std);

if isvector(perf), perf = perf(:); end
if isvector(perf_std), perf_std = perf_std(:); end

    %% ---- Helper functions ----
    function conf = calculate_confusion_matrix(normalize)
        % calculates the confusion matrix. if normalize = 1, the absolute
        % counts are normalized to fractions by dividing each row by the
        % row total
        
        conf = cell(sz_output);

        % Must compare each class with each other class, therefore create
        % all combinations of pairs of class labels
        comb = zeros(2, nclasses*2);
        for jj=1:nclasses
            comb(1,(jj-1)*nclasses+1:jj*nclasses) = 1:nclasses;
            comb(2,(jj-1)*nclasses+1:jj*nclasses) = jj;
        end
        
        % The confusion matrix is a nclasses x nclasses matrix where each
        % row corresponds to the true label and each column corresponds to 
        % the label predicted by the classifier . The (i,j)-th element of the
        % matrix specifies how often class i has been classified as class j
        % by the classifier. The diagonal of the matrix contains the 
        % correctly classified cases, all off-diagonal elements are
        % misclassifications.
        
        % confusion_fun calculates the confusion matrix but it is a bit
        % complicated so let's unpack it step-by-step, starting from the
        % inner brackets and working our way towards the outer brackets:
        % - To get the (ii,jj) element of the matrix, we compare how often
        %   the true label (lab) is equal to ii while at the same time the 
        %   predicted label (cfo) is equal to jj. This is then summed. The 
        %   code in the sum(...) function accomplishes this
        % - arrayfun(...) executes the sum function for every combination 
        %   of classes 
        % - reshape(...) turns the vector returned by arrayfun into a 
        %   [nclasses x nclasses] cell matrix
        confusion_fun = @(lab,cfo) reshape( arrayfun( @(ii,jj) sum(lab==ii & cfo==jj),comb(1,:),comb(2,:),'Un',0), nclasses, nclasses, []);
        
        % Looping across the extra dimensions if model_output is multi-dimensional
        for xx=1:nextra
            % Use cellfun to apply the function defined above to each cell
            tmp = cellfun(confusion_fun, y, model_output(dim_skip_token{:},xx), 'Un', 0);
            
            % We're almost done. We just need to transform the cell
            % matrices contained in each cell of tmp into ordinary
            % matrices. However, with multi-dimensional outputs (such as 
            % in mv_classify_timextime) we need to make sure that the 
            % resulting matrix has the correct dimensions, i.e.
            % [nclasses x nclasses x ...]
            tmp = cellfun( @(c) cellfun( @(x) reshape(x,1,1,[]), c,'Un',0), tmp, 'Un',0);
            
            % Now we're ready to transform each cell of tmp into a matrix 
            tmp = cellfun( @cell2mat, tmp, 'Un',0);
            
            if normalize 
                % confusion_fun gives us the absolute counts 
                % for each combination of classes. 
                % It is useful to normalize the confusion matrix such that the
                % cells represent proportions instead of absolute counts. To 
                % this end, each row is divided by the number of
                % samples in that row. As a result every row sums to 1.
                nor = cellfun( @(lab) 1./repmat(sum(lab,2), [1,nclasses]), tmp, 'Un',0);
                tmp = cellfun( @(lab,nor) lab .* nor, tmp, nor, 'Un', 0);

                % We get a pathological situation when one fold does not 
                % contain any trials of a particular class (nor gets Inf). It's
                % unclear how to deal with this situation because averaging
                % the confusion matrix across folds does not make so much sense 
                % any more. A perhaps reasonable fix is to set this column to
                % 0. 
                for c=1:numel(tmp)
                    tmp{c}(isinf(tmp{c})) = 0;
                end
            end

            conf(dim_skip_token{:},xx) = tmp;
        end
    end

    function precision = calculate_precision(conf)
        precision = cell(sz_output);
        if nclasses==2
            % for two classes we use the standard formula TP / (TP + FP)
            for xx=1:nextra % Looping across the extra dimensions if model_output is multi-dimensional
                precision(dim_skip_token{:},xx) = cellfun( @(c) c(1,1,:,:,:,:,:,:)./(c(1,1,:,:,:,:,:,:)+c(2,1,:,:,:,:,:,:)), conf(dim_skip_token{:},xx), 'Un',0);
            end
        else
            % for more than two classes, we use the generalisation from
            % Sokolava and Lapalme (2009) precision_i = c_ii / (sum_j c_ji) 
            % where c_ij = (i,j)-th entry of the unnormalized confusion matrix
            for xx=1:nextra % Looping across the extra dimensions if model_output is multi-dimensional
                % a nested arrayfun is needed for eg timextime
                % classification when the cell contents are
                % multi-dimensional too
                if ismatrix(conf{1,1,xx})
                    precision(dim_skip_token{:},xx) = cellfun( @(c) diag(c)./sum(c,1)', conf(dim_skip_token{:},xx), 'Un',0);
                else
                    % the content is a [classes x nclasses x ...]
                    % multidimensional array, so we need to iterate
                    % across the last dimension using a nested arrayfun
                    % and a nested cell2mat to convert it back to a
                    % matrix
                    precision(dim_skip_token{:},xx) = cellfun( @(c) cell2mat(arrayfun(@(ix) diag(c(:,:,ix))./sum(c(:,:,ix),1)', 1:(numel(c)/nclasses/nclasses),'Un',0)), conf(dim_skip_token{:},xx), 'Un',0);
                end
            end
        end
    end

    function recall = calculate_recall(conf)
        recall = cell(sz_output);

        if nclasses==2
            % for two classes we use the standard formula TP / (TP + FN)
            for xx=1:nextra % Looping across the extra dimensions if model_output is multi-dimensional
                recall(dim_skip_token{:},xx) = cellfun( @(c) c(1,1,:,:,:,:,:,:)./(c(1,1,:,:,:,:,:,:)+c(1,2,:,:,:,:,:,:)), conf(dim_skip_token{:},xx), 'Un',0);
            end
        else
            % for more than two classes, we use the generalisation from
            % Sokolava and Lapalme (2009) recall_i = c_ii / (sum_j c_ij) 
            % where c_ij = (i,j)-th entry of the unnormalized confusion matrix
            for xx=1:nextra % Looping across the extra dimensions if model_output is multi-dimensional
                if ismatrix(conf{1,1,xx})
                    recall(dim_skip_token{:},xx) = cellfun( @(c) diag(c)./sum(c,2), conf(dim_skip_token{:},xx), 'Un',0);
                else
                    % the content is a [classes x nclasses x ...]
                    % multidimensional array, so we need to iterate
                    % across the last dimension using a nested arrayfun
                    % and a nested cell2mat to convert it back to a
                    % matrix
                    recall(dim_skip_token{:},xx) = cellfun( @(c) cell2mat(arrayfun(@(ix) diag(c(:,:,ix))./sum(c(:,:,ix),2), 1:(numel(c)/nclasses/nclasses),'Un',0)), conf(dim_skip_token{:},xx), 'Un',0);
                end
            end
        end
    end
end

