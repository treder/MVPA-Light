function stat = mv_statistics(cfg, result, X, y)
% Performs statistical analysis on classification or regression results of
% the high-level functions such as mv_classify_across time and mv_regress.
%
% mv_statistics implements both Level 1 (subject-level) and Level 2 (group
% level) statistical analysis of MVPA results.
%
% Usage:
% stat = mv_statistics(cfg, result, X, y)
%
%Parameters:
% result       - struct describing the classification outcome. Can be
%                obtained as second output argument from functions
%                mv_crossvalidate, mv_classify_across_time,
%                mv_classify_timextime, mv_searchlight, mv_classify, and
%                mv_regress.
%                Level 1 analysis is performed if a single result struct is
%                provided.
%                Level 2 analysis is performed if multiple result structs
%                (corresponding to eg multiple subjects) are provided.
% X            - input data used to obtain result
% y            - input class labels or responses used to obtain result
%
% cfg is a struct with parameters:
% .test        - specify the statistical test that is applied.
%                Level 1 tests:
%                'binomial': binomial test (classification only) is
%                            performed on accuracy values. Requires a
%                            classification result using the accuracy metric
%                'permutation': permutation test calculates p-values by
%                               repeating the classification using shuffled
%                               class labels or repsonse values
%                Level 2 uses permutation tests.
% .alpha       - significance threshold (default 0.05)
% .metric      - if result contains multiple metrics, choose the target
%                metric (default [])
% .width       - width of progress bar in characters (default 20)
%
% Further details regarding specific tests:
%
% BINOMIAL:
% Uses a binomial distribution to calculate the p-value under the null
% hypothesis that classification accuracy = chance (typically 0.5)
% Treating results from cross-validation analysis: since the
% sum (which is the unnormalized mean) of binomially distributed variables
% is binomial, too, we can treat the results on the folds and repetitions
% as a single large binomial test. This is possible because the
% classification accuracy has been calculated using weighted averaging, and
% hence the total number of hits is equal to the average accuracy *  total
% number of samples.
% Additional parameters for binomial test:
% .chance      - specify chance level (default 0.5)
%
% PERMUTATION:
% Permutation testing is a non-parametric approach based on an empirical
% null-distribution obtained via permutations. To this end, the
% multivariate analysis is repeated many times (typically 1000's) with
% class labels/responses being randomly shuffled
% The classification or regression analysis is repeated many times using
% randomly shuffled class labels or responses.
% Additional parameters for permutation test:
% .n_permutations        - number of permutations (default 1000)
% .correctm              - correction applied for multiple comparisons
%                         'none'
%                         'bonferroni'
%                         'cluster'
% .tail                 - -1 or 1 (default = 1), specifies whether the
%                          lower tail (-1), or the upper tail (+1) is
%                          computed Typically, for accuracy
%                          measures such as AUC, precision/recall etc we
%                          set tail=1 since we want to test whether
%                          the performance metric is larger than expected
%                          by chance. Vice versa, for error metrics often
%                          used in regression (eg MSE, MAE), tail=-1 since
%                          we want to check whether the error is lower than
%                          expected. (two-tailed testing is current not
%                          supported)
% .keep_null_distribution - if 1, the full null distribution is saved
%                          in a matrix [n_permutations x (size of result)].
%                          Note that for multi-dimensional data this matrix
%                          can be very large (default 0)
%
% CLUSTER PERMUTATION TEST:
% For cluster-based multiple comparisons correction the procedure laid out
% in Maris & Oostenveld (2007) and implemented in FieldTrip is followed.
% Here, the classification or regression metrics serve as statistics that
% quantify the difference between experimental conditions. The following
% options determine how the metrics will be thresholded and combined into
% one statistical value per cluster.
% To use cluster permutation tests, set test = 'permutation' and correctm =
% 'cluster'. In addition to the parameters available for the permutation
% test, the following parameters control the behaviour of the cluster
% correction:
%
%   .clusterstatistic    - how to combine the single samples that belong to
%                          a cluster, 'maxsum', 'maxsize' (default =
%                          'maxsum'). This is the actual statistic for the
%                          cluster permutation test.
%   .clustercritval      - cutoff-value for thresholding (this parameter
%                          must be set by the user). For instance it could
%                          be 0.7 for classification accuracy so that all
%                          accuracy values >= 0.7 would be considered for
%                          clusters. If tail=0, a vector of two
%                          numbers must be provided (high and low cutoff).
%                          The exact numerical choice of the critical
%                          value is up to the user (see Maris & Oostenveld,
%                          2007, discussion).
%    .conndef            - 'minimal' or 'maximal', how neighbours are
%                          defined. Minimal means only directly
%                          neighbouring elements in a matrix are
%                          neighbours. Maximal means that also diagonally
%                          related elements are considered neighbours. E.g.
%                          in the matrix [1 2; 3 4] 1 and 4 are neighbours
%                          for conndef ='maximal' but not 'minimal'
%                          (default 'minimal'). Note that this requires the
%                          Image Processing Toolbox.
%   .neighbours          - in some cases the neighbourhood cannot be
%                          purely spatially (eg when one dimension encodes
%                          channels). A cell array of binary matrices can
%                          be used in this case. [TODO]
%                          (see mv_classify for details)
%
% LEVEL 2 PERMUTATION AND CLUSTER PERMUTATION TEST:
% Level 2 tests are tests at a group level. Typically, a set of MVPA
% results is available (imagine MVPA has been performed for 12 different
% participants). The goal of the Level 2 test is to establish whether an
% effect is significant at the group level.
% Both between-subject and within-subject designs are supported. The
% following parameters apply to both within- and between-subject designs:
% 
%   .design              - 'within' or 'between'
%   .statistic           - statistic used to measure the difference (within
%                          or between groups): 'ttest' 'wilcoxon' 'mean'
%                          If cluster correction is used, this statistic
%                          will be compared against the clustercritval.
%                          Between-subject design: the independent
%                          samples version of ttest/wilcoxon test is used.
%                          Within-subject design: the difference
%                          (between two conditions or between one condition
%                          and the null value) is compared against 0.
%                          clustercritval corresponds to t-values (ttest)
%                          or z-values (wilcoxon).
%   .clustercritval      - the meaning of clustercritval for a level 2 test
%                          is different from a level 1 test. In level 2, it
%                          refers to the critical t or z value (depending
%                          on whether ttest or Wilcoxon signrank is used).
%   
% Specific parameters for between-subjects design:
%   .group               - vector of 1's and 2's specifying which group
%                          each result struct belongs to
%
% Specific parameters for within-subject design: In within-subject designs,
% the metric itself can have two values (e.g. dvals for two classes). In
% this case the difference between these two values is calculated
% automatically. Alternatively, if only one metric is available, it has to 
% be compared against a null value (e.g. AUC compared against 0.5).
%   .null                - defines the null value for a within-subject
%                          design (e.g. 0.5 for AUC). If not specified it is
%                          assumed that the metric itself is
%                          two-dimensional, such as dval (default [])
%
% Returns:
% stat       - structure with description of the statistical result.
%              Important fields:
%                stat.p       - p-values
%                stat.mask    - logical significance mask (giving 1 when p < alpha)
%                stat.statistic - for Level 2 tests returns the raw statistic
%
% Reference:
% Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of
% EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177–190.
% https://doi.org/10.1016/j.jneumeth.2007.03.024


%Output:
% stat - struct with statistical output

% (c) Matthias Treder

mv_set_default(cfg,'alpha', 0.05);
mv_set_default(cfg,'metric', []);
mv_set_default(cfg,'feedback', 1);
mv_set_default(cfg,'width', 30);

mv_set_default(cfg,'chance', 0.5);

mv_set_default(cfg,'correctm', 'none');
mv_set_default(cfg,'n_permutations', 1000);
mv_set_default(cfg,'clusterstatistic', 'maxsum');
mv_set_default(cfg,'tail', 1);
mv_set_default(cfg,'keep_null_distribution', false);
mv_set_default(cfg,'conndef', 'minimal');
mv_set_default(cfg,'neighbours', []);

% Level 2 statistics settings
mv_set_default(cfg,'design', []);
mv_set_default(cfg,'statistic', 'ttest');
mv_set_default(cfg,'null', []);

% Level 1 or Level 2?
level = double(numel(result)>1) + 1;

%% Statistical testing
stat = struct('test',cfg.test,'alpha',cfg.alpha);

%  ---------------------------
%% --------- Level 1 ---------
%  ---------------------------
if level == 1
    switch(cfg.test)
        case 'binomial'
            %%% --- LEVEL 1 BINOMIAL ---
            if ~iscell(result.metric)
                metric = {result.metric};
                perf = {result.perf};
            else
                metric = result.metric;
                perf = result.perf;
            end
            ix = find(ismember(metric, {'acc' 'accuracy'}));
            if isempty(ix)
                error('Binomial test requires accuracy but the only available metric is %s', strjoin(metric))
            end
            perf = perf{ix};
            stat.chance = cfg.chance;
            
            % N is the total number of samples
            n_samples = result.n;
            
            % Calculate p-value using the cumulative distribution function, testing
            % H0: the observed accuracy was due to chance
            stat.p = 1 - binocdf( floor(perf * n_samples), n_samples, cfg.chance);
            
            % Create binary mask (1 = significant)
            stat.mask = stat.p < cfg.alpha;
            
        case 'permutation'
            %%% --- LEVEL 1 PERMUTATION ---
            result = select_metric(result);
            perf = result.perf;
            result.cfg.metric = result.metric;  % for MVPA
            result.cfg.feedback = 0;
            is_clustertest = strcmp(cfg.correctm, 'cluster');
            has_neighbour_matrix = ~isempty(cfg.neighbours);
            
            % some sanity checks
            if nargin<4, error('Data and class labels/responses need to be provided as inputs for permutation tests'); end
            if strcmp(cfg.correctm, 'cluster') && ~isfield(cfg, 'clustercritval')
                error('cfg.correctm=''cluster'' but cfg.clustercritval is not set')
            end
            
            % high-level function
            fun = eval(['@' result.function]);
            
            if cfg.feedback
                if strcmp(cfg.correctm, 'none'), cor = 'no'; else cor = cfg.correctm; end
                fprintf('Performing permutation test with %s correction for multiple comparisons.\n', cor);
            end
            
            if is_clustertest
                % Initialize cluster test: find initial clusters and calculate
                % cluster sizes. Keep it stored in vector
                conn = conndef(ndims(result.perf), cfg.conndef); % init connectivity type
                if cfg.tail == 1, C = (perf > cfg.clustercritval);
                else C = (perf < cfg.clustercritval);
                end
                CC_init = bwconncomp(C,conn);
                n_clusters = numel(CC_init.PixelIdxList);
                if n_clusters == 0; error('Found no clusters in input data. Consider changing clustercritval'), end
                if cfg.feedback, fprintf('Found %d clusters.\n', n_clusters); end
                
                real_clusterstat = compute_cluster_statistic(CC_init, perf, 0);
                % 2) after each permutation recalculate clusters and cluster values
                % and create a counts vector (a count for each original cluster)
                counts = zeros(size(real_clusterstat));
            else
                % Standard permutation test:
                % represents the histogram: counts how many times the permutation
                % statistic is more extreme that the reference values in perf
                counts = zeros(size(perf));
                if cfg.keep_null_distribution, null_distribution = zeros([cfg.n_permutations, size(perf)]); end
            end
            
            if cfg.feedback, fprintf('Running %d permutations ', cfg.n_permutations); end
            
            % run permutations
            for n=1:cfg.n_permutations
                
                % permute class labels/responses
                y_perm = y(randperm(result.n), :);
                
                % run mvpa with permuted data
                permutation_perf = fun(result.cfg, X, y_perm);
                if cfg.keep_null_distribution, null_distribution(n,:,:,:,:,:,:,:,:,:,:,:) = permutation_perf; end
                
                if is_clustertest
                    if cfg.tail == 1, C = (permutation_perf > cfg.clustercritval);
                    else C = (permutation_perf < cfg.clustercritval);
                    end
                    CC = bwconncomp(C,conn);
                    permutation_clusterstat = compute_cluster_statistic(CC, permutation_perf, 1);
                    if ~isempty(permutation_clusterstat)
                        if cfg.tail == 1
                            counts = counts + double(permutation_clusterstat > real_clusterstat);
                        else
                            counts = counts + double(permutation_clusterstat < real_clusterstat);
                        end
                    end
                else
                    % standard permutation test
                    if cfg.tail == 1
                        counts = counts + double(permutation_perf > perf);
                    else
                        counts = counts + double(permutation_perf < perf);
                    end
                end
                
                % update progress bar
                if cfg.feedback, mv_print_progress_bar(n, cfg.n_permutations, cfg.width); end
                
            end
            if cfg.feedback, fprintf('\n'); end
            
            % bonferroni correction of alpha value
            if strcmp(cfg.correctm, 'bonferroni')
                alpha = cfg.alpha / numel(result.perf);
            else
                alpha = cfg.alpha;
            end
            
            % calculate p-value and build mask
            stat.p = counts / cfg.n_permutations;
            if is_clustertest
                sig = find(stat.p < alpha);
                stat.mask = false(size(perf));
                stat.mask_with_cluster_numbers = zeros(size(perf));
                for ii=1:numel(sig)
                    stat.mask(CC_init.PixelIdxList{sig(ii)}) = true;
                    stat.mask_with_cluster_numbers(CC_init.PixelIdxList{sig(ii)}) = sig(ii);
                end
                stat.n_significant_clusters = numel(sig);
            else
                stat.mask = stat.p < alpha;
            end
            
            stat.alpha          = alpha;
            stat.correctm       = cfg.correctm;
            stat.n_permutations = cfg.n_permutations;
            if cfg.keep_null_distribution, stat.null_distribution = null_distribution; end
    end
    
else
    %  ---------------------------
    %% --------- Level 2 ---------
    %  ---------------------------
    switch(cfg.test)
    
        case 'permutation'
            %%% --- LEVEL 2 PERMUTATION ---
            n_results = numel(result);
            is_clustertest = strcmp(cfg.correctm, 'cluster');
            
            % select desired metric
            for n=1:n_results
                result{n} = select_metric(result{n});
            end
            % concatenate data from all subjects
            perf_all_subjects = concatenate_results(result);
            metric = result{1}.metric;
            
            % some sanity checks
            assert(~strcmp(cfg.correctm, 'cluster') || isfield(cfg, 'clustercritval'),'cfg.correctm = ''cluster'' but cfg.clustercritval is not set')
            assert(~isempty(cfg.design), 'You need to specify cfg.design')
            is_within = strcmp(cfg.design, 'within'); 
            
            if is_within
                % subtract either null value or subtract two dimensions
                % from each other to turn the problem from a paired-samples
                % to a one-sample test
                if ~isempty(cfg.null)
                    if cfg.feedback, fprintf('Subtracting null value %2.2f.\n', cfg.null); end
                    perf_all_subjects = perf_all_subjects - cfg.null;
                else
                    ix = find(ismember(result{1}.perf_dimension_names, 'metric'));
                    if isempty(ix) || numel(ix)>2 || size(result{1}.perf, ix) ~= 2
                        error('cfg.null = [] but metric %s does not seem to have exactly two dimensions', metric)
                    else
                        before = repmat({':'}, [1 ix-1]);
                        after = repmat({':'}, [1 ndims(result{1}.perf)-ix]);
                        % subtract the two dimensions of the metric from
                        % each other so we can perform a one-sample test
                        % later on
                        for n=1:n_results
                            p = result{n}.perf;
                            p = p(before{:}, 1, after{:}) - p(before{:}, 2, after{:});
                            result{n}.perf = p;
                        end
                        perf_all_subjects = concatenate_results(result);
                    end
                end
            end
            
            if cfg.feedback
                if strcmp(cfg.correctm, 'none'), cor = 'no'; else cor = cfg.correctm; end
                fprintf('Performing level 2 permutation test with %s correction and %s statistic.\n', cor, upper(cfg.statistic));
            end
            
            % Calculate statistic for real data
            if is_within
                perf = within_subject_statistic(cfg.statistic, perf_all_subjects);
            else
                perf = between_subjects_statistic(cfg.statistic, perf_all_subjects, cfg.group);
            end
            
            if is_clustertest
                % Initialize cluster test: find initial clusters and calculate
                % cluster sizes. Keep it stored in vector
                conn = conndef(ndims(result{1}.perf), cfg.conndef); % init connectivity type
                if cfg.tail == 1, C = (perf > cfg.clustercritval);
                else C = (perf < cfg.clustercritval);
                end
                CC_init = bwconncomp(C,conn);
                n_clusters = numel(CC_init.PixelIdxList);
                if n_clusters == 0; error('Found no clusters in input data. Consider changing clustercritval'), end
                if cfg.feedback, fprintf('Found %d clusters.\n', n_clusters); end
                
                real_clusterstat = compute_cluster_statistic(CC_init, perf, 0);
                % 2) after each permutation recalculate clusters and cluster values
                % and create a counts vector (a count for each original cluster)
                counts = zeros(size(real_clusterstat));
            else
                % Standard permutation test:
                % represents the histogram: counts how many times the permutation
                % statistic is more extreme that the reference values in perf
                counts = zeros(size(result{1}.perf));
                if cfg.keep_null_distribution, null_distribution = zeros([cfg.n_permutations, size(result{1}.perf)]); end
            end
            
            if cfg.feedback, fprintf('Running %d permutations ', cfg.n_permutations); end
            
            % run permutations
            for n=1:cfg.n_permutations
                % Permute data and calculate statistic
                if is_within
                    % Permutation for within-subject design: 
                    % randomly reverse the sign of the perf for a subject
                    permutation_perf_all_subjects = bsxfun(@times, perf_all_subjects, sign(randn(n_results, 1)));
                    permutation_perf = within_subject_statistic(cfg.statistic, permutation_perf_all_subjects);
                else
                    % Permutation for between-subjects design: 
                    % Randomly permute the group
                    permutation_group = cfg.group(randperm(length(cfg.group)));
                    permutation_perf = between_subjects_statistic(cfg.statistic, perf_all_subjects, permutation_group);
                end
                if cfg.keep_null_distribution, null_distribution(n,:,:,:,:,:,:,:,:,:,:,:) = permutation_perf; end
                
                if is_clustertest
                    if cfg.tail == 1, C = (permutation_perf > cfg.clustercritval);
                    else C = (permutation_perf < cfg.clustercritval);
                    end
                    CC = bwconncomp(C,conn);
                    permutation_clusterstat = compute_cluster_statistic(CC, permutation_perf, 1);
                    if ~isempty(permutation_clusterstat)
                        if cfg.tail == 1
                            counts = counts + double(permutation_clusterstat > real_clusterstat);
                        else
                            counts = counts + double(permutation_clusterstat < real_clusterstat);
                        end
                    end
                else
                    % standard permutation test
                    if cfg.tail == 1
                        counts = counts + double(permutation_perf > perf);
                    else
                        counts = counts + double(permutation_perf < perf);
                    end
                end
                
                % update progress bar
                if cfg.feedback, mv_print_progress_bar(n, cfg.n_permutations, cfg.width); end
                
            end
            if cfg.feedback, fprintf('\n'); end
            
            % bonferroni correction of alpha value
            if strcmp(cfg.correctm, 'bonferroni') 
                alpha = cfg.alpha / numel(perf);
            else
                alpha = cfg.alpha;
            end
            
            % calculate p-value and build mask
            stat.p = counts / cfg.n_permutations;
            if is_clustertest
                sig = find(stat.p < alpha);
                stat.mask = false(size(perf));
                stat.mask_with_cluster_numbers = zeros(size(perf));
                for ii=1:numel(sig)
                    stat.mask(CC_init.PixelIdxList{sig(ii)}) = true;
                    stat.mask_with_cluster_numbers(CC_init.PixelIdxList{sig(ii)}) = sig(ii);
                end
                stat.n_significant_clusters = numel(sig);
            else
                stat.mask = stat.p < alpha;
            end
            
            stat.alpha          = alpha;
            stat.correctm       = cfg.correctm;
            stat.n_permutations = cfg.n_permutations;
            stat.statistic      = perf;
            if cfg.keep_null_distribution, stat.null_distribution = null_distribution; end
    end 
end

%% -- helper functions --
    function res = select_metric(res)
        % selects the metric specified in cfg.metric from result
        if isempty(cfg.metric)
            if iscell(res.perf) && numel(res.perf) > 1
                error('Multiple metrics available (%s), you need to set cfg.metric to select one', strjoin(res.metric))
            end
        elseif ischar(res.metric) && ~strcmp(res.metric, cfg.metric)
            error('Metric %s requested but only %s available', cfg.metric, res.metric)
        elseif iscell(res.metric)
            ix = (ismember(res.metric, cfg.metric));
            if any(ix)
                res.perf = res.perf{ix};
                res.metric = res.metric{ix};
                res.perf_dimension_names = res.perf_dimension_names{ix};
            else
                error('Metric %s requested but the available metrics are: %s', cfg.metric, strjoin(res.metric))
            end
        end
    end

    function perf = concatenate_results(result)
        % Given multiple result structs, adds a leading dimension and
        % appends all perf matrices
        perf = zeros([numel(result), size(result{1}.perf)]);
        for j=1:n_results
            perf(j,:,:,:,:,:,:,:,:,:,:,:,:,:) = result{j}.perf;
        end
    end

    function perf_stat = within_subject_statistic(statistic, cperf)
        % calculates one-sample statistic along the first dimension. 
        switch(statistic)
            case 'mean', perf_stat = mean(cperf, 1);
            case 'ttest', [~,~,~,sts] = ttest(cperf); perf_stat = sts.tstat;
            case 'wilcoxon', perf_stat = mv_stat_wilcoxon_signrank(cperf);
        end
        perf_stat = squeeze(perf_stat);
    end

    function perf_stat = between_subjects_statistic(statistic, cperf, group)
        % calculates two-samples independent samples statistic along the first dimension. 
        switch(statistic)
            case 'mean', perf_stat = mean(cperf(group==1,:), 1) - mean(cperf(group==2,:), 1);
            case 'ttest', [~,~,~,sts] = ttest2(cperf(group==1,:),cperf(group==2,:)); perf_stat = sts.tstat;
            case 'wilcoxon', perf_stat = mv_stat_wilcoxon_ranksum(cperf, group);
        end
        perf_stat = squeeze(perf_stat);
    end

    function clusterstat = compute_cluster_statistic(CC, P, max_only)
        % Compute statistic for the cluster permutation test
        %     max_only - if 1 returns only the cluster statistic for the
        %                largest cluster
        switch(cfg.clusterstatistic)
            case 'maxsize'
                clusterstat = cellfun(@numel, CC.PixelIdxList);
            case 'maxsum'
                clusterstat = cellfun(@(ix) sum(P(ix)), CC.PixelIdxList);
        end
        if max_only
            clusterstat = max(clusterstat);
        end
    end
end