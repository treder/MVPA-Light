function [perf, perf_std] = mv_calculate_performance(metric, output_type, cf_output, clabel, dim)
%Calculates a classifier performance metric such as classification accuracy
%based on the classifier output (e.g. labels or decision values). In
%cross-validation, the metric needs to be calculated on each test fold
%separately and then averaged across folds, since a different classifier
%has been trained in each fold.
%
%Usage:
%  [perf, perf_std] = mv_classifier_performance(metric, cf_output, clabel, dim)
%
%Parameters:
% metric            - desired performance metric:
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
%                     classifier output
% output_type       - type of classifier ('clabel', 'dval', or 'prob'). See
%                     mv_get_classifier_output for details
% cf_output         - vector of classifier outputs (labels or dvals). If
%                     multiple test sets have been validated using
%                     cross-validation, a (possibly mult-dimensional)
%                     cell array should be provided with each cell
%                     corresponding to one test set.
% clabel            - vector of true class labels. If multiple test sets
%                     have been validated using cross-validation, a cell
%                     array of labels should be provided (same size as
%                     cf_output)
% dim               - index of dimension across which values are averaged
%                     (e.g. dim=2 if the second dimension is the number of
%                     repeats of a cross-validation). Default: [] (no
%                     averaging).
%                     Note that a weighted mean is used, that is, folds
%                     are weighed proportionally to their number of test
%                     samples.
%
%Note: cf_output is typically a cell array. The following functions provide
%classifier output as multi-dimensional cell arrays:
% - mv_crossvalidate: 2D [repeats x K] cell array
% - mv_classify_across_time: 3D [repeats x K x time points] cell array
% - mv_classify_timextime: 3D [repeat x K x time points] cell array
%
% In all three cases, however, the corresponding clabel array is just
% [repeats x K] since class labels are repeated for all time points. If 
% cf_output has more dimensions than clabel, the clabel array is assumed 
% to be identical across the extra dimensions.
%
% Each cell should contain a [testlabel x 1] vector of classifier outputs
% (ie labels or dvals) for the according test set. Alternatively, it can
% contain a [testlabel x ... x ...] matrix, like the output of
% mv_classify_timextime. But then all other dimensions need to have the
% same size.
%
%Returns:
% perf     - performance metric
% perf_std - standard deviation of the performance metric across folds 
%            and repetitions, indicating the 
%            variability of the metric.

% (c) Matthias Treder 2017

if nargin<5
    dim=[];
end

if ~iscell(cf_output), cf_output={cf_output}; end
if ~iscell(clabel), clabel={clabel}; end

% Check the size of the cf_output and clabel. nExtra keeps the number of
% elements in the extra dimensions if ndims(cf_output) > ndims(clabel). For
% instance, if we classify across time, cf_output is [repeats x folds x time]
% and clabel is [repeats x folds] so we have 1 extra dimension (time).
sz_cf_output = size(cf_output);
nExtra = prod(sz_cf_output(ndims(clabel)+1:end));
% dimSkipToken helps us looping across the extra dimensions
dimSkipToken = repmat({':'},[1, ndims(clabel)]);

% For some metrics dvals or probabilities are required
if strcmp(output_type,'clabel') && any(strcmp(metric,{'dval' 'roc' 'auc'}))
    error('To calculate dval/roc/auc, classifier output must be given as dvals or probabilities, not as class labels')
end

perf = cell(sz_cf_output);

% Calculate the requested performance metric
switch(metric)
    
    case {'acc', 'accuracy'}
        %%% ------ ACC: classification accuracy -----
        
        if strcmp(output_type,'clabel')
            % Compare predicted labels to the true labels. To this end, we
            % create a function that compares the predicted labels to the
            % true labels and takes the mean of the comparison. This gives
            % us the classification performance for each test fold.
            fun = @(cfo,lab) mean(bsxfun(@eq,cfo,lab(:)));
        elseif strcmp(output_type,'dval')
            % We want class 1 labels to be positive, and class 2 labels to
            % be negative, because then their sign corresponds to the sign
            % of the dvals. To this end, we transform the labels as
            % clabel =  -clabel + 1.5 => class 1 will be +0.5 now, class 2
            % will be -0.5
            % We first need to transform the classifier output into labels.
            % To this end, we create a function that multiplies the the
            % dvals by the true labels - for correct classification the
            % product is positive, so compare whether the result is > 0.
            % Taking the mean of this comparison gives classification
            % performance.
            fun = @(cfo,lab) mean(bsxfun(@times, cfo, -lab(:)+1.5) > 0);
        elseif strcmp(output_type,'prob')
            % Probabilities represent the posterior probability for class
            % being class 1, ranging from 0 to 1. To transform 
            % probabilities into class labels, subtract 0.5 from the
            % probabilities and also transform the labels (see previous
            % paragraph about dvals for details)
            fun = @(cfo,lab) mean(bsxfun(@times, cfo-0.5, -lab(:)+1.5) > 0);
        end
        
        % Looping across the extra dimensions if cf_output is multi-dimensional
        for xx=1:nExtra
            % Use cellfun to apply the function defined above to each cell
            perf(dimSkipToken{:},xx) = cellfun(fun, cf_output(dimSkipToken{:},xx), clabel, 'Un', 0);
        end
        
    case 'dval'
        %%% ------ DVAL: average decision value for each class ------
        
        perf = cell([sz_cf_output,2]);
        
        % Aggregate across samples, for each class separately
        if nExtra == 1
            perf(dimSkipToken{:},1) = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:),1), cf_output, clabel, 'Un',0);
            perf(dimSkipToken{:},2) = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:),1), cf_output, clabel, 'Un',0);
        else
            for xx=1:nExtra % Looping across the extra dimensions if cf_output is multi-dimensional
                perf(dimSkipToken{:},xx,1) = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:),1), cf_output(dimSkipToken{:},xx), clabel, 'Un',0);
                perf(dimSkipToken{:},xx,2) = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:),1), cf_output(dimSkipToken{:},xx), clabel, 'Un',0);
            end
        end
        
    case 'tval'
        %%% ------ TVAL: independent samples t-test values ------
        % Using the formula for unequal sample size, equal variance: 
        % https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes.2C_equal_variance
        perf = cell(sz_cf_output);
        
         % Aggregate across samples, for each class separately
        if nExtra == 1
            % Get means
            M1 = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:),1), cf_output, clabel, 'Un',0);
            M2 = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:),1), cf_output, clabel, 'Un',0);
            % Variances
            V1 = cellfun( @(cfo,lab) nanvar(cfo(lab==1,:,:,:,:,:)), cf_output, clabel, 'Un',0);
            V2 = cellfun( @(cfo,lab) nanvar(cfo(lab==2,:,:,:,:,:)), cf_output, clabel, 'Un',0);
            % Class frequencies
            N1 = cellfun( @(lab) sum(lab==1), clabel, 'Un',0);
            N2 = cellfun( @(lab) sum(lab==2), clabel, 'Un',0);
            % Pooled standard deviation
            SP = cellfun( @(v1,v2,n1,n2) sqrt( ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)  ), V1,V2,N1,N2, 'Un',0);
            % T-value
            perf = cellfun( @(m1,m2,n1,n2,sp) (m1-m2)/(sp*sqrt(1/n1 + 1/n2)) , M1,M2,N1,N2,SP,'Un',0);
        else
            for xx=1:nExtra % Looping across the extra dimensions if cf_output is multi-dimensional
                % Get means
                M1 = cellfun( @(cfo,lab) nanmean(cfo(lab==1,:,:,:,:,:),1), cf_output(dimSkipToken{:},xx), clabel, 'Un',0);
                M2 = cellfun( @(cfo,lab) nanmean(cfo(lab==2,:,:,:,:,:),1), cf_output(dimSkipToken{:},xx), clabel, 'Un',0);
                % Variances
                V1 = cellfun( @(cfo,lab) nanvar(cfo(lab==1,:,:,:,:,:)), cf_output(dimSkipToken{:},xx), clabel, 'Un',0);
                V2 = cellfun( @(cfo,lab) nanvar(cfo(lab==2,:,:,:,:,:)), cf_output(dimSkipToken{:},xx), clabel, 'Un',0);
                % Class frequencies
                N1 = cellfun( @(lab) sum(lab==1), clabel, 'Un',0);
                N2 = cellfun( @(lab) sum(lab==2), clabel, 'Un',0);
                % Pooled standard deviation
                SP = cellfun( @(v1,v2,n1,n2) sqrt( ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)  ), V1,V2,N1,N2, 'Un',0);
                % T-value
                perf(dimSkipToken{:},xx) = cellfun( @(m1,m2,n1,n2,sp) (m1-m2)./(sp.*sqrt(1/n1 + 1/n2)) , M1,M2,N1,N2,SP,'Un',0);
            end
        end
        
    case 'auc'
        %%% ----- AUC: area under the ROC curve -----
        
        % AUC can be calculated by sorting the dvals, traversing the
        % positive examples (class 1) and counting the number of negative
        % examples (class 2) with lower values
        
        %         [cf_output,soidx] = sort(cf_output,'descend');
        %
        %         sz= size(cf_output);
        %         perf= zeros([1, sz(2:end)]);
        %
        %         % Calculate AUC for each column of cf_output
        %         for ii=1:prod(sz(2:end))
        %             clabel = clabel(soidx(:,ii));
        %
        %             % Find all class indices
        %             isClass1Idx = find(clabel(:)== 1);
        %             isClass2 = (clabel(:)==2 );
        %
        %             % Count number of False Positives with lower value
        %             for ix=1:numel(isClass1Idx)
        %                 perf(1,ii) = perf(1,ii) + sum(isClass2(isClass1Idx(ix)+1:end));
        %             end
        %
        %             % Correct by number of True Positives * False Positives
        %             perf(1,ii) = perf(1,ii)/ (numel(isClass1Idx) * sum(isClass2));
        %         end
        
        % There is different ways to calculate AUC. An efficient one for
        % our purpose is to sort the clabel vector in an ascending fashion,
        % the dvals are later sorted accordingly using soidx.
        [clabel, soidx] = cellfun(@(lab) sort(lab,'ascend'), clabel, 'Un',0);
        N1 = cellfun( @(lab) {sum(lab==1)}, clabel);
        N2 = cellfun( @(lab) {sum(lab==2)}, clabel);
        
        % Anonymous function gtfun takes a specific row cii and checks how many
        % times the value (corresponding to class label 1) in one column is greater than the value in the
        % rows of the submatrix cn1 (corresponding to the class labels 2). Ties are weighted by 0.5. This gives a
        % count for each column separately.
        % cii and cn1 represent the possibly matrix-sized content of a cell
        % of cf_output
        gtfun = @(cii, cn1) (sum(bsxfun(@gt, cii, cn1) + 0.5*bsxfun(@eq,cii,cn1) ));
        % arrsum repeats gtfun for each sample in matrix c corresponding to
        % class 1. The counts are then summed across the class 1 samples
        arrsum =  @(c,n1) sum(cell2mat(   arrayfun(@(ii) gtfun(c(ii,:,:,:),c(n1+1:end,:,:,:)), 1:n1,'Un',0)'    ) );
        for xx=1:nExtra
            % Sort decision values using the indices of the sorted labels.
            % Add a bunch of :'s to make sure that we preserve the other
            % (possible) dimensions
            cf_so = cellfun(@(c,so) c(so,:,:,:,:,:) , cf_output(dimSkipToken{:},xx), soidx, 'Un',0);
            % Use cellfun to perform the following operation within each
            % cell:
            % For each class 1 sample, we count how many class 2 exemplars
            % have a lower decision value than this sample. If there is a
            % tie (dvals equal for two exemplars of different classes),
            % we add 0.5. Dividing that number by the #class 1 x #class 2
            % (n1*n2) gives the AUC.
            % We do this by applying the above-defined arrsum to every cell
            % in cf_output and then normalising by (n1*n2)
            perf(dimSkipToken{:},xx) = cellfun(@(c,n1,n2) arrsum(c,n1)/(n1*n2), cf_so, N1,N2, 'Un',0);
        end
        
    case 'confusion'
        %%% ---------- confusion matrix ---------
        
        if ~isClassLabel
            error('confusion matrix requires class labels as classifier output')
        end
        
        nclasses = max(max(max( vertcat(clabel{:}) )));
%         perf = cell([sz_cf_output,nclasses,nclasses]);

        % Must compare each class with each other class, therefore create
        % all combinations of pairs of class labels
        comb = combvec(1:nclasses,1:nclasses);
        
        % The confusion matrix is a nclasses x nclasses matrix where each
        % row corresponds to the label predicted by the classifier and each 
        % column corresponds to the true label. The (i,j)-th element of the
        % matrix specifies how often class j has been classified as class i
        % by the classifier. The diagonal of the matrix contains the 
        % correctly classified cases, all off-diagonal elements are
        % misclassifications.
        
        % confusion_fun calculates the confusion matrix but it is a bit
        % complicated so let's unpack it step-by-step, starting from the
        % inner brackets and working our way towards the outer brackets:
        % - To get the (ii,jj) element of the matrix, we compare how often
        %   the predicted label (cfo) is equal to ii while at the same time 
        %   the true label (lab) is equal to jj. This is then summed. The 
        %   code in the sum(...) function accomplishes this
        % - arrayfun(...) executes the sum function for every combination 
        %   of classes 
        % - reshape(...) turns the vector returned by arrayfun into a 
        %   [nclasses x nclasses] cell matrix
        confusion_fun = @(cfo,lab) reshape( arrayfun( @(ii,jj) sum(cfo==ii & lab==jj),comb(1,:),comb(2,:),'Un',0), nclasses, nclasses, []);
        
        % Looping across the extra dimensions if cf_output is multi-dimensional
        for xx=1:nExtra
            % Use cellfun to apply the function defined above to each cell
            tmp = cellfun(confusion_fun, cf_output(dimSkipToken{:},xx), clabel, 'Un', 0);
            
            % We're almost done. We just need to transform the cell
            % matrices contained in each cell of tmp into ordinary
            % matrices. However, with multi-dimensional outputs (such as 
            % in mv_classify_timextime) we need to make sure that the 
            % resulting matrix has the correct dimensions, i.e.
            % [nclasses x nclasses x ...]
            tmp = cellfun( @(c) cellfun( @(x) reshape(x,1,1,[]), c,'Un',0), tmp, 'Un',0);
            
            % Now we're ready to transform each cell of tmp into a matrix 
            tmp = cellfun( @cell2mat, tmp, 'Un',0);
            
            % confusion_fun gives us the absolute counts 
            % for each combination of classes. 
            % It is useful to normalise the confusion matrix such that the
            % cells represent proportions instead of absolute counts. To 
            % this end, each c-th column is divided by the number of
            % samples in that column. As a result every column sums to 1.
            nor = cellfun( @(lab) 1./repmat(sum(lab,1), [nclasses,1]), tmp, 'Un',0);
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
            
            perf(dimSkipToken{:},xx) = tmp;
        end
        
    otherwise, error('Unknown metric: %s',cfg.metric)
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

%% Average the performance metric 

% Average across requested dimensions
for nn=1:numel(dim)

    % Calculate standard error [use unweighted averages here]
    if nn==1
        if size(perf, dim(nn)) == 1 && numel(dim)>1
            % dimension dim(nn) has size = 1, so there is no std here. We
            % must then take the std over the other dimension
            % second dimension. This can happen when there is only 1
            % repetition.
            perf_std = nanstd(perf, [], dim(nn+1));
        else
            perf_std = nanstd(perf, [], dim(nn));
        end
    else
        perf_std = mean(perf_std, dim(nn));
    end
    
    % For averaging, use a WEIGHTED mean: Since some test sets may have
    % more samples than others (since the number of data points is not
    % always integer divisible by K), folds with more test samples give
    % better estimates. They should be weighted higher proportionally to
    % the number of samples.
    % To achieve this, first multiply the statistic for each fold with the
    % number of samples in this fold. Then sum up the statistic and divide
    % by the total number of samples.
    if nn==1
        num_samples = cellfun(@numel,clabel);
        
        % Multiply metric in each fold with its number of samples
        perf = bsxfun(@times, perf, num_samples);
    end
    
    % Sum metric and respective number of samples
    perf = nansum(perf, dim(nn));
    num_samples = nansum(num_samples, dim(nn));
    
    % Finished - we need to normalise again by the number of samples to get
    % back to the original scale
    if nn==numel(dim)
        perf = perf ./ num_samples;
    end
end

perf = squeeze(perf);
perf_std = squeeze(perf_std);

