% UNDERSTANDING PREPROCESSING
% 
% In this example we look into preprocessing pipelines. Preprocessing
% includes demeaning, z-scoring, PCA, sample averaging, feature
% extraction methods such as PCA, and any other
% approaches that operate on the data prior to training.
% 
% An important distinction is between what will be referred to as 
% 'global' vs 'nested' preprocessing: In 'global' preprocessing, an
% operation is applied to the whole dataset (including both training and
% test data) at once before classification is done. In some cases this can 
% lead to overfitting. For instance, consider the Common Spatial Patterns
% (CSP) approach which makes use of the class labels.
% Training a CSP filter on the whole dataset means there is information
% transfer between train set and test set: The components used in the
% train set have been derived by including information from the test set.
% 
% Nested preprocessing avoids this by obtaining the parameters for an
% operation solely from the train data. The parameters (e.g. principal 
% components) extracted from the train set are then applied to the test 
% set. This assures that no information from the test set went into the 
% preprocessing of the train data.
%
% Contents:
% (1) Global preprocessing
% (2) Using mv_preprocess functions for global preprocessing
% (3) Using mv_preprocess functions for nested preprocessing
% (4) Nested preprocessing with parameters
% (5) Preprocessing pipelines
% (6) Preprocessing pipelines with parameters
% (7) Preprocessing for regression
%
% Note: If you are new to working with MVPA-Light, make sure that you
% complete the introductory tutorials first:
% - getting_started_with_classification
% - getting_started_with_regression
% They are found in the same folder as this tutorial.

close all
clear

% Load data
[dat,clabel] = load_example_data('epoched2', 0); % 0 = don't perform z-scoring of the data for now
X = dat.trial;

%% (1) Global preprocessing
% In 'global' preprocessing, the full dataset is preprocessed once before
% the analysis is started. Global preprocessing should be used with
% caution to make sure that there is no information transfer between train
% and test set. 
%
% A very useful global preprocessing operation is z-scoring, whereby the
% data is scaled and centered at 0. It is recommended in general, but it is
% important for classifiers such as Logistic Regression and SVM for numerical
% reasons. Furthermore, z-scoring brings all features on the same footing,
% irrespective of the units they were measured in originally.

% Let's start by looking at the feature/time point with the largest mean
% value
max(max(mean(X,1)))

% zscore across samples dimension
X = zscore(X, [], 1); 

% double check that the mean is now (very close to) zero
max(max(mean(X,1)))

%% (2) Using mv_preprocess functions for global preprocessing
% MVPA-Light features a preprocessing framework that was designed for
% nested preprocessing, but it can also be used for global preprocessing.
% We will perform z-scoring using the framework.
%
% Currently available preprocessing functions can be found in 
% https://github.com/treder/MVPA-Light/tree/master/preprocess

% Preprocessing function have hyperparameters, let us start by getting the
% default parameters
pparam = mv_get_preprocess_param('zscore')
% pparam.is_train_set is useful in nested preprocessing (since the functions
% operate differently on train and test sets) and it is usually used
% automatically by high-level functions such as mv_classify
% pparam.dimension specifies the dimension across which z-scoring is done.
% The default value 1 is what we want. 
% Let's reload the non-z-scored data and perform z-scoring again:
[dat,clabel] = load_example_data('epoched2', 0);
X = dat.trial;
[~, X] = mv_preprocess_zscore(pparam, X);

% confirm that the z-scoring was successful
max(max(mean(X,1)))

%%%%%% EXERCISE 1 %%%%%%
% Z-scoring performs both demeaning and rescaling of the data, but
% sometimes we may want to demean the data without rescaling. Use
% mv_preprocess_demean to accomplish this.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (3) Using mv_preprocess functions for nested preprocessing
% Now we explore how to perform nested preprocessing. For z-scoring, the
% nested preprocessing operation works as follows: on the training data,
% the means and standard deviations are calculated along the first
% dimension, and the train data are z-scored accordingly. For the test
% data, the same mean and standard deviation are used to center and scale
% the data. 

[dat,clabel] = load_example_data('epoched2', 0);
X = dat.trial;

% For comparative purposes, let us first perform classification without
% preprocessing.
cfg = [];
cfg.metric          = 'acc';
cfg.classifier      = 'svm';
cfg.repeat          = 1;
cfg.k               = 10;
perf = mv_classify_across_time(cfg, X, clabel);

% We can add nested preprocessing by setting the preprocess field:
cfg.preprocess      = 'zscore';
perf2 = mv_classify_across_time(cfg, X, clabel);

figure
plot(dat.time, perf)
hold all
plot(dat.time, perf2)
legend({'Without zscore' 'With nested zscore'})
grid on
% The classification accuracy is very similar, but it took much longer to
% train SVM with non-normalized data.

%%%%%% EXERCISE 2 %%%%%%
% From the results we can see that SVM has a bias (it has >50% performance
% even in the pre-stimulus baseline). This is because the classes are
% imbalanced. To correct this, use the 'undersample' preprocessing
% function. It undersamples the training data such that both classes have
% equal proportions. You can use data that was globally z-scored by loading 
% the data without the second argument:
% [dat,clabel] = load_example_data('epoched2'); % data has been z-scored
%%%%%%%%%%%%%%%%%%%%%%%%

%% (4) Nested preprocessing with parameters
[dat,clabel] = load_example_data('epoched2');
X = dat.trial;
% So far we used nested preprocessing with the default parameters. What if
% we want to change one of the parameters? This is accomplished using the 
% .preprocess_param field. We will use the average_samples preprocessing
% function to this end. Samples from
% the same class are split into multiple groups and then the dataset is
% replaced by the group means. This reduces the data and at the same time
% increases SNR. 

% First, let us familiarize ourselves with average_samples by looking at the
% default paramters and running it in a 'global' way
pparam = mv_get_preprocess_param('average_samples')

[~, X_averaged, clabel_averaged] = mv_preprocess_average_samples(pparam, X, clabel);

% We can see that X and clabel are down from 310 samples to just 61
% samples. This is because average_samples splits the data into groups of
% samples and then calculates the averages within each group - the average
% then becomes the new averaged sample. How many samples go into each group
% is controlled by the group_size parameter. The default is 5 which is why
% we end up with 310 / 5 = 61 samples. 
size(X)
size(X_averaged)

%%%%%% EXERCISE 3 %%%%%%
% Perform sample averaging again, but this time form groups of 2 samples.
% The result should contain 310 / 2 = 155 samples.
%%%%%%%%%%%%%%%%%%%%%%%%

% Now let us perform sample averaging in a nested fashion. Since we are not
% defining the group size here, the default size of 5 will be used.
cfg = [];
cfg.preprocess      = 'average_samples';
perf5 = mv_classify_across_time(cfg, X, clabel); % classification with group size 5

% To define the group size, we have to add the .preprocess_param field:
cfg.preprocess_param = [];
cfg.preprocess_param.group_size = 2;
perf2 = mv_classify_across_time(cfg, X, clabel); % classification with group size 2

figure
plot(dat.time, perf5), hold all, plot(dat.time, perf2)
legend({'group size 5' 'group size 2'})
% group size 5 yields a better performance than group size 2, since more
% samples are used for averaging

%%%%%% EXERCISE 4 %%%%%%
% average_samples should only be used with linear classifiers since
% averaging itself is a linear operation. For nonlinear classification
% using kernel classifiers, average_kernel can be used instead (see 
% help of mv_preprocess_average_kernel for a motivation). 
% Train a SVM classifier with a RBF kernel using average kernel with a
% group size of 4.
% Hint: use compute_kernel_matrix to precompute the kernel first and then
% set the kernel hyperparameter to 'precomputed'. Precomputing kernel
% matrices is also covered in the advanced_classification tutorial.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (5) Preprocessing pipelines
% Multiple preprocessing operations can be chained together to form
% preprocessing pipelines. Let us perform z-scoring first followed by
% sample averaging. This can be realised easily realized by providing a
% cell array of operations 
cfg = [];
cfg.preprocess      = {'zscore' 'average_samples'};
[~, result] = mv_classify_across_time(cfg, X, clabel);

mv_plot_result(result, dat.time)

%%%%%% EXERCISE 5 %%%%%%
% What if you reverse the order of the two preprocessing operations, do you
% expect the same result?
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% EXERCISE 6 %%%%%%
% Add PCA to the preprocessing pipeline. At which location in the pipeline
% do you add it?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (6) Preprocessing pipelines with parameters
% To get fine control over the preprocessing pipeline from the previous 
% section, we will set the parameters for the different preprocessing
% steps. Earlier we've seen that this can be achieved using the
% cfg.preprocess_param field. In case of a preprocessing pipeline with
% multiple preprocessing operations, multiple structs with parameters need
% to be provided.
% Let's first set the preprocessing pipeline
cfg = [];
cfg.preprocess      = {'zscore' 'average_samples'};

% We now want to set the group size of average samples to 4.
% We start by initializing preprocess_param as an empty cell array.
cfg.preprocess_param = {};
% We do not need to provide any parameters to zscore. average_samples is 
% the second operation, so we have to initialize the 2nd cell of the array.
cfg.preprocess_param{2} = [];
% We can now set the group size parameter in this 2nd cell. 
cfg.preprocess_param{2}.group_size = 4;

perf = mv_classify_across_time(cfg, X, clabel);


% There is an alternative notation for setting parameters using a cell array
% instead of a struct. This makes the notation slightly shorter.
cfg.preprocess_param = {};
cfg.preprocess_param{2} = {'group_size' 4};

perf = mv_classify_across_time(cfg, X, clabel);

% To summarize, this is the full definition of the cfg struct using the
% cell array notation:
cfg = [];
cfg.preprocess              = {'zscore' 'average_samples'};
cfg.preprocess_param        = {};
cfg.preprocess_param{2}     = {'group_size' 4};

perf = mv_classify_across_time(cfg, X, clabel);

%%%%%% EXERCISE 7 %%%%%%
% Add PCA to the preprocessing pipeline, after zscore and before
% average_samples. Set the number of Principal Components (PCs) to 10.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (7) Preprocessing for regression
% The examples so far only considered classification. Preprocessing can be
% defined in the same way when using the mv_regress function.
% However, not all preprocessing operations apply to regression. For instance,
% average_samples and average_kernel uses the class labels to form groups
% and hence are not useful for regression. However, other functions such as
% zscore and PCA can be applied for both classification and regression.

% Let's start with the simulated regression dataset from the beginning of
% getting_started_with_regression:
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

% As an example, let's perform nested preprocessing with z-score.
cfg = [];
cfg.model       = 'ridge';
cfg.metric      = {'mae' 'r_squared'};
cfg.preprocess  = 'zscore';
[perf, result] = mv_regress(cfg, X, y);

mv_plot_result(result)

%%%%%% EXERCISE 8 %%%%%%
% Perform regression using zscore following by PCA with 15 Principal
% Components.
%%%%%%%%%%%%%%%%%%%%%%%%

% Congrats, you finished the tutorial!

%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
% Let us get the default parameters of the demean function
pparam = mv_get_preprocess_param('demean')

% Perform demeaning 
[dat,clabel] = load_example_data('epoched2', 0);
X = dat.trial;

% check the scaling/standard deviation
max(max(std(X,[],1)))

% apply preprocessing
[~, X] = mv_preprocess_demean(pparam, X);

% we can confirm that the data is demeaned but not rescaled
max(max(mean(X,1)))
max(max(std(X,[],1)))

%% SOLUTION TO EXERCISE 2
[dat,clabel] = load_example_data('epoched2');
X = dat.trial;

cfg = [];
cfg.metric          = 'acc';
cfg.classifier      = 'svm';
cfg.repeat          = 1;
cfg.k               = 10;
% Simply set preprocess to 'undersample'
cfg.preprocess      = 'undersample';
perf3 = mv_classify_across_time(cfg, X, clabel);

% we can see now that SVM's prediction performance is around 50% in the
% pre-stimulus baseline, as expected
plot(dat.time, perf3)

%% SOLUTION TO EXERCISE 3
pparam = mv_get_preprocess_param('average_samples');
pparam.group_size = 2;  % define the group size here

[~, X_averaged, clabel_averaged] = mv_preprocess_average_samples(pparam, X, clabel);

%% SOLUTION TO EXERCISE 4 
% Let us first check the parameters of average_kernel
mv_get_preprocess_param('average_kernel')

% Precompute RBF kernel
cfg = [];
cfg.kernel              = 'rbf';
cfg.gamma               = .1;
cfg.regularize_kernel   = 0.001;
X_kernel = compute_kernel_matrix(cfg, X);

% Train
cfg = [];
cfg.classifier              = 'svm';
cfg.hyperparameter          = [];
cfg.hyperparameter.kernel   = 'precomputed';
cfg.k                       = 10;
cfg.repeat                  = 1;
cfg.preprocess              = 'average_kernel';
cfg.preprocess_param        = [];
cfg.preprocess_param.group_size = 2;

perf = mv_classify_across_time(cfg, X_kernel, clabel);

figure
plot(dat.time, perf)

%% SOLUTION TO EXERCISE 5
% If operations are carried out in a different order, the results can
% be different. In this case, the results are quite similar.
cfg = [];
cfg.preprocess      = {'average_samples' 'zscore'};
[~, result] = mv_classify_across_time(cfg, X, clabel);

mv_plot_result(result, dat.time)

%% SOLUTION TO EXERCISE 6
% There is no right or wrong location at which to insert the PCA. One could
% argue that it makes sense to perform PCA before the averaging, since the
% covariance matrix can then be estimated using more samples. Also,
% z-scoring is typically a basic preprocessing step that precedes other
% preprocessing operations. Therefore the preprocessing pipeline zscore ->
% PCA -> average_samples is chosen here, but a different ordering can be
% useful, too.
cfg = [];
cfg.preprocess      = {'zscore' 'pca' 'average_samples'};
[~, result] = mv_classify(cfg, X, clabel);

mv_plot_result(result, dat.time)

%% SOLUTION TO EXERCISE 7
% Using the help we see that the n parameter specifies the number of PCs
help mv_preprocess_pca

cfg = [];
cfg.preprocess              = {'zscore' 'pca' 'average_samples'};
cfg.preprocess_param        = {};
cfg.preprocess_param{2}     = {'n' 10}; % PCA is the 2nd operation therefore use the 2nd cell
cfg.preprocess_param{3}     = {'group_size' 4}; % average_samples is the 3rd operation

perf = mv_classify_across_time(cfg, X, clabel);

%% SOLUTION TO EXERCISE 8
% Recreate the data first
X = simulate_erp_peak(n_trials, n_time_points, pos, width, amplitude, weight, [], scale);
y = amplitude + 0.5 * randn(n_trials, 1);

cfg = [];
cfg.model               = 'ridge';
cfg.metric              = {'mae' 'r_squared'};
cfg.preprocess          = {'zscore' 'pca'};
cfg.preprocess_param    = {};
cfg.preprocess_param{2} = {'n' 15};

[perf, result] = mv_regress(cfg, X, y);

mv_plot_result(result)