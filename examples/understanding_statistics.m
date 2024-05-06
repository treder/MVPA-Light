% UNDERSTANDING STATISTICS
%
% In the introductory tutorials, various metrics are calculated for
% classification and regression problems. In many neuroimaging applications, 
% we also want to quantify the statistical significance of these metrics.
%
% To this end, we look into how to perform statistical testing of MVPA data. 
% The main purpose is to generate p-values and get an idea whether the
% classification or regression performance is significantly better than
% chance.
%
% Contents:
% (1) Introduction to mv_statistics
% (2) Single-subject classification: binomial test
% (3) Single-subject classification: permutation test
% (4) Single-subject classification: cluster permutation test
% (5) Single-subject regression: permutation test
% (6) Group statistics: generate and prepare data
% (7) Group statistics: within-subject cluster permutation test for AUC
% (8) Group statistics: within-subject cluster permutation test for DVAL
% (9) Group statistics: create a patient group dataset
% (10) Group statistics: between-subjects cluster permutation test for AUC
%
% Note: If you are new to working with MVPA-Light, make sure that you
% complete the introductory tutorials first:
% - getting_started_with_classification
% - getting_started_with_regression
% They are found in the same folder as this tutorial.

close all
clear

% Load data
[dat,clabel] = load_example_data('epoched2');
X = dat.trial;

%% (1) Introduction to mv_statistics
% In MVPA-Light, significance can be tested uing the function
% mv_statistics. It returns p-values associated with the classification
% or regression results and it has methods for correcting for multiple
% comparisons.
% For more details on the available tests, see the help of mv_statistics
help mv_statistics
% Looking at the signature of the function, we see that it expects a result
% struct as one of the input parameters. Statistical testing of MVPA
% results is really a two stage process: in the first stage, an analysis
% using a high-level function (eg mv_classify) is performed and the result
% struct is obtained. In the second stage, the result struct is passed on
% to mv_statistics along with the data (X and clabel/y) and a cfg struct
% defining the details of the statistical test.

% In statistical analysis we can differentiate between single-subject and
% group statistics. In single-subject statistics (also called level 1)
% we investigate whether the classification result for a single subject is 
% statistically significant (different from chance level). In group 
% statistics (also called level 2), we want to establish whether the 
% classification result is statistically significant at the group level. 

%% (2) Single-subject statistics: binomial test for accuracy
% We start by running an ordinary classification across time analysis.
% Since we want to perform a binomial test below, we need to select 
% classification accuracy as a metric. This will give us 131 accuracy
% values (one for each time point).
rng(21)
cfg =  [];
cfg.repeat          = 2;
cfg.metric          = 'accuracy'; % binomial test needs the accuracy metric
[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);

mv_plot_result(result, dat.time)

% Now that we calculated the metric, we can pass the result on to
% mv_statistics.
% The main input to the mv_statistics function is the cfg struct and the
% classification result struct. We can request a binomial test by setting
% the cfg.test field
cfg = [];
cfg.test    = 'binomial';

stat = mv_statistics(cfg, result)

% stat is a structure that contains the result of the statistical analysis.
% It contains a number of fields:
%   stat.test: the name of the test
%   stat.alpha: significance threshold
%   stat.chance: chance level (50%)
% Since we didn't set alpha and the chance level the default values were
% used.

% The field stat.p contains 131 p-values, one for each of the time points.
% Note that no multiple comparisons correction has been performed.
stat.p

% stat.mask is a binary array that marks the time points that are
% statistically significant
stat.mask

% It can be used with the mv_plot_result
% function in order to define a mask to be applied to the data
mv_plot_result(result, dat.time, 'mask', stat.mask)

% Comparing the plot to the plot without mask, we can see that the
% significant time points have been marked bold. There seems to be a big
% cluster from 0 to 0.8 s where the classification accuracy is significant. 
% Unfortunately, some pre-stimulus time points are significant, too.  

%%%%%% EXERCISE 1 %%%%%%
% The classes are imbalanced. Class 2 is the majority class, it makes up 
% 65.81% of the samples. Change the chance level of the test to account 
% for this. Also chance alpha to 0.1.
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% EXERCISE 2 %%%%%%
% The same statistical test can be applied to higher-dimensional data such 
% as time generalization data or time-frequency data.
% Use mv_classify_timextime to perform time generalization, then run 
% the binomial test again. What is the effect of stat.mask when plotting?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (3) Single-subject statistics: permutation test
% A permutation test is a nonparametric test that works with any metric in
% principle. It operates by shuffling the class labels and then rerunning
% the analysis many times. This is computationally much more intensive than
% the binomial test, but has the advantage of making less assumptions and
% working for any metric, not just classification accuracy.

% Let us rerun the classification across time using AUC
cfg =  [];
cfg.repeat          = 1;         % set to 1 to speed up the permutation test below
cfg.metric          = 'auc';

[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);

% Specify the permutation test and the number of permutations. The number
% of permutations defines how many times the classification is repeated
% with shuffled class labels. mv_statistics extracts all the parameters and
% hyperparameters used in the classification from the result struct, so no
% further manual settings are required.
% (this analysis will take a few minutes - watch the progress bar)
cfg = [];
cfg.test            = 'permutation';
cfg.n_permutations  = 500;

stat_permutation = mv_statistics(cfg, result, dat.trial, clabel);

% If we plot the result we find a similar pattern as for the binomial test.
% Again, some pre-stimulus time points (around -0.25) are significant. When 
% performing cluster correction (next section), we will see that this
% pre-stimulus effect disappears, suggesting that it was arose due to the lack of
% correction for multiple comparisons.
mv_plot_result(result, dat.time, 'mask', stat_permutation.mask)

%%%%%% EXERCISE 3 %%%%%%
% Perform the permutation test again, but use F1 score as a
% metric. To be finished more quickly, you can set n_permutations to 100.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (4) Single-subject statistics: cluster permutation test
% The permutation test does not solve the multiple comparisons problem that
% we encounter when we test multiple time points. mv_statistics implements
% the cluster permutation test introduced by Maris & Oostenveld:
%   Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of
%   EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177–190.
%   https://doi.org/10.1016/j.jneumeth.2007.03.024

% Starting from a permutation test, we can perform a cluster permutation
% test by setting correctm='cluster'. See Maris & Oostenveld, the FieldTrip
% documentation, or the help of mv_statistics for details on the other
% parameters. 
% (this will take a few minutes again)
cfg = [];
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.clustercritval  = 0.6;
cfg.n_permutations  = 500;

stat_cluster = mv_statistics(cfg, result, dat.trial, clabel);

% When we plot the result we can see that the effect in the pre-stimulus
% period disappears
mv_plot_result(result, dat.time, 'mask', stat_cluster.mask)

%%%%%% EXERCISE 4 %%%%%%
% Similar to Exercise 1, perform a cluster permutation test for a time
% generalization result (using mv_classify_timextime). Use 100 permutations
% only to speed up the analysis. 
% What do you notice when plotting the result?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (5) Single-subject regression: permutation test
% So far we used mv_statistics for classification. It can also be used to
% investigate regression results.
% Let us recreate the ERP dataset used in getting_started_with_regression:
n_trials = 300;
time = linspace(-0.2, 1, 201);
n_time_points = numel(time);
pos = 100; 
width = 10;
amplitude = 3*randn(n_trials,1) + 3;
weight = abs(randn(64, 1));
scale = 1;
X = simulate_erp_peak(n_trials, n_time_points, pos, width, amplitude, weight, [], scale);
y = amplitude + 0.5 * randn(n_trials, 1);

% Perform regression across time
cfg = [];
cfg.metric      = 'mae';
cfg.model       = 'ridge';
cfg.repeat      = 2;
[~, result_ridge] = mv_regress(cfg, X, y);

% In our simulated data we can predict y from the ERP in the range 0.2-0.6
mv_plot_result(result_ridge, time)

% Now let's establish whether the effect in 0.2-0.6 s is statistically
% significant.
% Normally in classification metrics we look for values that are larger
% than expected by chance. For MAE and MSE, the opposite is true, we look
% for values that are smaller than expected. We need to set cfg.tail = -1
% to make sure we calculate the p-value using the left side of the tail 
cfg = [];
cfg.test            = 'permutation';
cfg.n_permutations  = 100;
cfg.tail            = -1;
stat_permutation = mv_statistics(cfg, result_ridge, X, y);

% Plotting the result we see that we have a significant effect in the range
% 0.2-0.6, just as expected
mv_plot_result(result_ridge, time, 'mask', stat_permutation.mask)

%%%%%% EXERCISE 5 %%%%%%
% Rerun the analysis using MSE instead of MAE.
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% EXERCISE 6 %%%%%%
% Can you run a cluster correction by adapting the approach from section 4
% for regression?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (6) Group statistics: generate and prepare data
% We will now move on from single-subject statistics to group statistics.
% In group statistics, we are only interested whether a classification
% effect is significant across the whole group. For meaningful group
% statistics, we should ideally have ~12 or more subjects. Since we have
% example data for only 3 subjects, we will create an artificial dataset
% here using simulated ERP data.

n_electrodes = 64;
n_subjects = 12;
n_trials = [60 80]; % number of trials in class 1 and 2
Xs = cell(n_subjects, 1);
clabels = cell(n_subjects, 1);

% We will again use the ERP generation function. This time we will generate
% two sets of ERPs, one for each of two classes. We will simulate a single
% ERP component whose mean amplitude differs between the two classes. In
% addition, we will simulate between-subject variability (some variability
% in the mean position and mean amplitude of the ERPs, as well as the
% weights that project the ERP source into the electrodes) and 
% within-subject variability (trial to trial variability in amplitude and
% position).
time = linspace(-0.2, 1, 201);
n_time_points = numel(time);

rng(41)
grand_mean_ERP_position = 100;
grand_mean_weights = abs(randn(n_electrodes, 1));
grand_mean_amplitude_class1 = 80;
grand_mean_amplitude_class2 = 60;
noise_scale = 1;

mean_amplitudes = cell(n_subjects, 1);

% Create data
for subject = 1:n_subjects
    % between subject variability
    mean_ERP_position = grand_mean_ERP_position + round(randn*5);
    mean_amplitudes{subject} = [grand_mean_amplitude_class1 + round(randn*8);
                      grand_mean_amplitude_class2 + round(randn*8)];
    weight = grand_mean_weights + 0.1 * randn(n_electrodes, 1);
    
    X_tmp = cell(2,1);
    
    for c = 1:2 % classes
        n = n_trials(c);
        % within-subject variability
        pos = mean_ERP_position + round(randn(n,1)*5);          % position of ERP peak in each trial
        width = 15 + round(randn(n,1));         % width of ERP peak in samples
        amplitude = mean_amplitudes{subject}(c) + randn(n,1)*3; % ERP amplitude
        X_tmp{c} = simulate_erp_peak(n, n_time_points, pos, width, amplitude, weight, [], noise_scale);
    end
    
    Xs{subject} = cat(1, X_tmp{:}); % concatenate data from both classes
    clabels{subject} = [ones(n_trials(1),1); 2*ones(n_trials(2),1)];
end

% plot a single trial from the first subject
figure
plot(time, squeeze(Xs{1}(104,10,:)))
xlabel('Time'), ylabel('Amplitude'), title('Trial 104, channel 10')

% Perform single subject MVPA
results = cell(n_subjects, 1);
for subject = 1:n_subjects
    fprintf('-- Subject %d --\n', subject)
    cfg = [];
    cfg.metric      = {'acc' 'auc' 'f1' 'precision' 'recall', 'dval'};
    cfg.k           = 10;
    cfg.repeat      = 2;
    [~, results{subject}] = mv_classify_across_time(cfg, Xs{subject}, clabels{subject});
end
% the results cell array contains the MVPA results for every subject. 
results{5} % e.g. results for 5-th subject

% Let's plot the AUC result for two subjects. We can see that the 
% classification performance differs across subjects. Since we only want to
% plot AUC (not any of the other metrics) we use mv_select_result to select
% the desired metric
mv_plot_result(mv_select_result(results{1},'auc'), time),title('Subject 1')
mv_plot_result(mv_select_result(results{2},'auc'), time),title('Subject 2')

% We can also plot all results in one plot. This plot will be a bit crowded
% since it will contain one line for every subject. The plot also shows
% that there is small variations in the position of the ERP peak across
% subjects.
result_merge = mv_combine_results(results, 'merge');
result_merge = mv_select_result(result_merge, 'auc'); % select AUC only
mv_plot_result(result_merge)

% Instead of plotting all subjects together, we can also calculate the
% grand average across subjects and plot this. To this end, we only need to
% replace 'merge' by 'average' when calling mv_combine_results. Note that
% the shaded area is now the standard deviation across subjects.
result_average = mv_combine_results(results, 'average');
result_average = mv_select_result(result_average, 'auc');
mv_plot_result(result_average)

%% (7) Group statistics: within-subject cluster permutation test for AUC
% For each subject and every time point, we have calculated AUC values. We
% will now perform a cluster permutation test. See Maris & Oostenveld's
% paper or the FieldTrip tutorials (https://www.fieldtriptoolbox.org/tutorial/cluster_permutation_timelock/) 
% for explanations on cluster permutation tests
cfg_stat = [];
cfg_stat.metric          = 'auc';
cfg_stat.test            = 'permutation';
cfg_stat.correctm        = 'cluster';  % correction method is cluster
cfg_stat.n_permutations  = 1000;

% Clusterstatistic is the actual statistic used for the clustertest.
% Normally the default value 'maxum' is used, we are setting it here
% explicitly for clarity. Maxsum adds up the statistics calculated at each
% time point (the latter are set below using cfg_stat.statistic)
cfg_stat.clusterstatistic = 'maxsum';
cfg_stat.alpha           = 0.05; % use standard significance threshold of 5%

% Level 2 stats design: we have to choose between within-subject and
% between-subjects. Between-subjects is relevant when there is two
% different experimental groups (eg patients vs controls) and we want to
% investigate whether their MVPA results are significantly different. Here,
% we have only one group and we want to see whether the AUC is
% significantly different from a null value, hence the statistical design
% is within-subject
cfg_stat.design          = 'within';
% cfg_stat.statistic defines how the difference between the AUC values and
% the null is calculated at each time point (across subjects). 
% We can choose t-test or its nonparametric counterpart Wilcoxon test. We
% choose Wilcoxon here.
cfg_stat.statistic       = 'wilcoxon';
% The null value for AUC (corresponding to a random classifier) is 0.5
cfg_stat.null            = 0.5;
% clustercritval is a parameter that needs to be selected by the user. In a
% Level 2 (group level) test, it represents the critical cutoff value for
% the statistic. Here, we selected Wilcoxon, so clustercritval corresponds
% to the cutoff value for the z-statistic which is obtained by a normal
% approximation.
cfg_stat.clustercritval  = 1.96;
% z-val = 1.65 corresponds to uncorrected p-value = 0.1
% z-val = 1.96 corresponds to uncorrected p-value = 0.05
% z-val = 2.58 corresponds to uncorrected p-value = 0.01
% Note that these z-values correspond to a two-sided test. In many cases it
% suffices to use a one-sided test. In this case, the following cutoff
% values are obtained:
% z-val = 1.282 corresponds to uncorrected p-value = 0.1
% z-val = 1.645 corresponds to uncorrected p-value = 0.05
% z-val = 2.326 corresponds to uncorrected p-value = 0.01

stat_level2 = mv_statistics(cfg_stat, results);

% plot the grand average result again and indicate the cluster in bold
mv_plot_result(result_average, time, 'mask', stat_level2.mask)

%%%%%% EXERCISE 7 %%%%%%
% Repeat the analysis using a T-test instead of Wilcoxon for the statistic.
% Is there any difference between the results? Is there any reason to
% select Wilcoxon over a T-test?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (8) Group statistics: within-subject cluster permutation test for DVAL
% We tested AUC values against a null value of 0.5 which we needed to
% define. A different approach could be to use decision values: since we
% already have two sets of dvals (one set for class 1, another for class 2),
% we do not need a separate null value, we just need to compare whether the
% dvals for the two classes are significantly different.
% We will use the cfg_stat struct from
% the previous section and just adapt the values that need to be changed.
cfg_stat.metric              = 'dval';

% By setting the null value to [], mv_statistics will assume that the
% metric itself is two-dimensional (which is the case for dval) and it will
% test the class 1 dvals against class 2 dvals.
cfg_stat.null           = [];

stat_level2 = mv_statistics(cfg_stat, results);

% calculate grand average dval and show significant cluster as a bold line
dval_average = mv_combine_results(results, 'average');
dval_average = mv_select_result(dval_average, 'dval');
mv_plot_result(dval_average, time, 'mask', stat_level2.mask)

%% (9) Group statistics: create a patient group dataset
% Let us imagine that the 12 subjects we used so far are actually a control
% group in a clinical study. We have an additional group of 11 patients who
% did the same study and what we are interested in is whether there is a
% significant difference in decodability between patients vs controls. 

controls = Xs;   % we call the original subjects controls now
n_controls = n_subjects;
results_controls = results;

% This requires a between-subjects statistical approach. Let us first
% create a second simulated dataset representing the 11 patients. We will
% use mostly the same code as berore, we only decrease the grand mean
% amplitude for class 1. This can simulate e.g. depression
% of ERP amplitude. Additionally we increase the noise from 1 to 1.5.
n_patients = 11;
patients = cell(n_patients, 1);
clabels_patients = cell(n_patients, 1);

grand_mean_amplitude_class1 = 64;  % this line is changed from 80 (controls) to simulate a depressed ERP amplitude 
grand_mean_amplitude_class2 = 60;
noise_scale = 1.5; % changed from 1 to 1.5: patient data is assumed to be more noisy

mean_amplitudes = cell(n_patients, 1);

% Create patient data
for subject = 1:n_patients
    % between subject variability
    mean_ERP_position = grand_mean_ERP_position + round(randn*5);
    mean_amplitudes{subject} = [grand_mean_amplitude_class1 + round(randn*3);
                      grand_mean_amplitude_class2 + round(randn*3)];
    weight = grand_mean_weights + 0.1 * randn(n_electrodes, 1);
    X_tmp = cell(2,1);
    for c = 1:2 % classes
        n = n_trials(c);
        pos = mean_ERP_position + round(randn(n,1)*5);          % position of ERP peak in each trial
        width = 15 + round(randn(n,1));         % width of ERP peak in samples
        amplitude = mean_amplitudes{subject}(c) + randn(n,1)*3; % ERP amplitude
        X_tmp{c} = simulate_erp_peak(n, n_time_points, pos, width, amplitude, weight, [], noise_scale);
    end
    
    patients{subject} = cat(1, X_tmp{:});
    clabels_patients{subject} = [ones(n_trials(1),1); 2*ones(n_trials(2),1)];
end

% Perform single subject MVPA for the patients
results_patients = cell(n_patients, 1);
for subject = 1:n_patients
    fprintf('-- Patient %d --\n', subject)
    cfg = [];
    cfg.metric      = {'acc' 'auc' 'f1' 'precision' 'recall', 'dval'};
    cfg.k           = 10;
    cfg.repeat      = 2;
    [~, results_patients{subject}] = mv_classify_across_time(cfg, patients{subject}, clabels_patients{subject});
end

% Let's have a look at the grand average result. We can indeed see that the
% peak classification performance is lower for patients than controls.
close all
result_average_patients = mv_combine_results(results_patients, 'average');
result_average_patients = mv_select_result(result_average_patients, 'auc');
result_average_patients.name = 'patients';
mv_plot_result(result_average_patients)

result_average_controls = mv_combine_results(results_controls, 'average');
result_average_controls = mv_select_result(result_average_controls, 'auc');
result_average_controls.name = 'controls';
mv_plot_result(result_average_controls)

% We can also combine both patients and controls into a single plot which
% makes it easier to compare
controls_and_patients_average = mv_combine_results({result_average_controls, result_average_patients}, 'merge');
mv_plot_result(controls_and_patients_average)

%% (10) Group statistics: between-subjects cluster permutation test for AUC
% Using the controls and patients datasets from the previous section, we
% will now perform a between-subjects cluster permutation test.
% First, set up the cfg struct. Most of the parameters are defined in
% the same way as before for the within-subject design
cfg_stat = [];
cfg_stat.metric          = 'auc';
cfg_stat.test            = 'permutation';
cfg_stat.correctm        = 'cluster';
cfg_stat.n_permutations  = 1000;
cfg_stat.statistic       = 'wilcoxon';
cfg_stat.null            = 0.5;
cfg_stat.clustercritval  = 1.96;
% z-val = 1.65 corresponds to p-value = 0.1
% z-val = 1.96 corresponds to p-value = 0.05
% z-val = 2.58 corresponds to p-value = 0.01

% There is parameters that need special attention when setting up a
% between-subjects analysis. The design needs to be set to 'between'
cfg_stat.design          = 'between';

% We will pool the controls and patients into a single cell array which is
% passed to mv_statistics. To know which subjects belongs to which group,
% we also need to specify the .group field: we have 12 subjects in group 1
% (controls) followed by 11 subjects in group 2 (patients). Note that it
% does not matter which group is defined as group 1 and which one as
% group 2.
all_results = [results_controls; results_patients];
cfg_stat.group = [ones(n_controls,1); 2*ones(n_patients,1)];

% Let's run the analysis
stat_between = mv_statistics(cfg_stat, all_results);

% Plot the result - the bold lines highlight the time interval corresponding
% to the significant cluster
mv_plot_result(controls_and_patients_average, time, 'mask', stat_between.mask)

%%%%%% EXERCISE 8 %%%%%%
% Repeat the analysis using 2D time x time data (time generalization). 
% Hint: you will need to re-run the MVPA for patients and controls.
%%%%%%%%%%%%%%%%%%%%%%%%

% Congrats, you finished the tutorial!

%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
% Let's reproduce the result first
cfg =  [];
cfg.repeat          = 2;
cfg.metric          = 'accuracy'; % binomial test needs the accuracy metric
[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);


% Chance level and alpha can be controlled using the respective fields of
% the cfg struct
cfg = [];
cfg.test    = 'binomial';
cfg.alpha   = 0.1;
cfg.chance  = 0.6581;

stat = mv_statistics(cfg, result)

% Plotting the result we see that the significant cluster is much smaller
% now
mv_plot_result(result, dat.time, 'mask', stat.mask)


%% SOLUTION TO EXERCISE 2
% We do not need to change much, only replace mv_classify_across_time by
% mv_classify_timextime
cfg =  [];
cfg.repeat          = 2;
cfg.metric          = 'accuracy';
[~, result] = mv_classify_timextime(cfg, dat.trial, clabel);

mv_plot_result(result, dat.time)

cfg = [];
cfg.test    = 'binomial';
stat = mv_statistics(cfg, result);

% The effect of stat.mask is 'masking out' (seting to 0) all parts of the
% data that are not significant. We can also observe that there is some
% significant time-time combinations in the pre-stimulus phase, which is
% probably a spurious result due to the lack of multiple comparisons
% correction
mv_plot_result(result, dat.time, 'mask', stat.mask)

%% SOLUTION TO EXERCISE 3
% We only need to change the metric when running mv_classify_across_time.
% For mv_statistics, we only reduce the number of permutations to 100 for
% performance reasons.
cfg =  [];
cfg.repeat          = 1;
cfg.metric          = 'f1';
[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);

cfg = [];
cfg.test    = 'permutation';
cfg.n_permutations = 100;

stat_permutation = mv_statistics(cfg, result, dat.trial, clabel);

mv_plot_result(result, dat.time, 'mask', stat_permutation.mask)

%% SOLUTION TO EXERCISE 4
% Again, we do not need to change much, only replace mv_classify_across_time 
% by mv_classify_timextime
cfg =  [];
cfg.repeat          = 1;         % set to 1 to speed up the permutation test below
cfg.metric          = 'auc';
[~, result] = mv_classify_timextime(cfg, dat.trial, clabel);

% (this will take a few minutes again)
cfg = [];
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.clustercritval  = 0.6;
cfg.n_permutations  = 100;

stat_cluster = mv_statistics(cfg, result, dat.trial, clabel);

% When we plot the result we notice that there is much less spurious
% activity in the pre-stimulus period (eg compare Exercise 1)
mv_plot_result(result, dat.time, 'mask', stat_cluster.mask)

%% SOLUTION TO EXERCISE 5
cfg = [];
cfg.metric      = 'mse'; % we only need to change this
cfg.model       = 'ridge';
cfg.repeat      = 2;
[~, result] = mv_regress(cfg, X, y);

cfg = [];
cfg.test            = 'permutation';
cfg.n_permutations  = 100;
cfg.tail            = -1;
stat_permutation = mv_statistics(cfg, result, X, y);

mv_plot_result(result, time, 'mask', stat_permutation.mask)

%% SOLUTION TO EXERCISE 6
% We can use almost the same code as for classification. There's only two
% things that need to be changed:
% - set cfg.tail = -1
% - clustercritval now refers to the cutoff in terms of MAE. It needs to be
%   set by hand. It's set to 1 here, but this is somewhat subjective.
cfg = [];
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.tail            = -1;
cfg.clustercritval  = 1;
cfg.n_permutations  = 100;

stat_cluster = mv_statistics(cfg, result_ridge, X, y);

mv_plot_result(result_ridge, time, 'mask', stat_cluster.mask)

%% SOLUTION TO EXERCISE 7
% Let's start by recreating the cfg struct
cfg_stat = [];
cfg_stat.metric          = 'auc';
cfg_stat.test            = 'permutation';
cfg_stat.correctm        = 'cluster';  % correction method is cluster
cfg_stat.n_permutations  = 1000;
cfg_stat.clusterstatistic = 'maxsum';
cfg_stat.alpha           = 0.05; % use standard significance threshold of 5%
cfg_stat.design          = 'within';
cfg_stat.statistic       = 'wilcoxon';
cfg_stat.null            = 0.5;
cfg_stat.clustercritval  = 1.96;

% We only need to change wilcoxon to ttest and then call the function
% again.
cfg_stat.statistic       = 'ttest';

stat_level2 = mv_statistics(cfg_stat, results);

mv_plot_result(result_average, time, 'mask', stat_level2.mask)
% We can see that the result looks very similar, so it does not seem to
% make a big difference which test statistic we use. On the other hand,
% t-test assumes that the data is normally distributed, which might not
% be true for AUC. The Wilcoxon signrank test does not make such 
% assumptions so could be preferred in this situation.

%% SOLUTION TO EXERCISE 8
% We start by calculating the time x time classification results for
% patients and controls.

% Perform single subject MVPA for the patients
results_patients = cell(n_patients, 1);
for subject = 1:n_patients
    fprintf('-- Patient %d --\n', subject)
    cfg = [];
    cfg.metric      = 'auc';
    cfg.k           = 10;
    cfg.repeat      = 2;
    [~, results_patients{subject}] = mv_classify_timextime(cfg, patients{subject}, clabels_patients{subject});
end

% Perform single subject MVPA for the controls
results_controls = cell(n_subjects, 1);
for subject = 1:n_controls
    fprintf('-- Control %d --\n', subject)
    [~, results_controls{subject}] = mv_classify_timextime(cfg, controls{subject}, clabels{subject});
end

% Before running the statistics, let's look at the data.
% Create average results for each group, then put both groups together in
% one struct
result_average_patients = mv_combine_results(results_patients, 'average');
result_average_patients.name = 'patients';

result_average_controls = mv_combine_results(results_controls, 'average');
result_average_controls.name = 'controls';

controls_and_patients_average = mv_combine_results({result_average_controls, result_average_patients}, 'merge');
mv_plot_result(controls_and_patients_average, time, time)

% For the statistical test, we use exatly the same settings as before
cfg_stat = [];
cfg_stat.metric          = 'auc';
cfg_stat.test            = 'permutation';
cfg_stat.correctm        = 'cluster';
cfg_stat.n_permutations  = 500;
cfg_stat.statistic       = 'wilcoxon';
cfg_stat.null            = 0.5;
cfg_stat.clustercritval  = 1.96;
cfg_stat.design          = 'between';

all_results = [results_controls;results_patients];
cfg_stat.group = [ones(n_controls,1); 2*ones(n_patients,1)];

% Rerun the analysis
stat_between = mv_statistics(cfg_stat, all_results);

% Plot the result again, this time highlighting the parts belonging to the
% significant cluster. All other parts of the image are masked out.
mv_plot_result(controls_and_patients_average, time, 'mask', stat_between.mask)

% We can also plot the raw Wilcoxon statistic masked by the selected
% clusters (showing only the significant parts). To this end, we use
% mv_plot_2D. For nicer layout we provide the time vectors for x and y axes
% as well as titles.
figure
mv_plot_2D(stat_between.statistic, 'mask', stat_between.mask, 'x',time, 'y',time,...
    'colorbar_title', cfg_stat.statistic, 'title','Masked statistic (control vs patient)')
