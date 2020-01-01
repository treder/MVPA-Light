# MVPA-Light
Matlab toolbox for classification and regression of multi-dimensional data

### News

* (November 2019) added [`mv_regress`](mv_regress.m) for regression of multi-dimensional data, and [ridge](#ridge) and [kernel ridge](#kernelridge) models. Furthermore, [`mv_statistics`](statistics/mv_statistics.m) now includes permutation and cluster permutation tests
* (August 2019) added [`mv_classify`](#mvclassify) for classification of multi-dimensional datasets (e.g. time-frequency), a [Naive Bayes classifier](#naivebayes) and the [kappa metric](#kappa)
* (July 2019) added [preprocessing module](#preprocessing) + precomputed kernels for [SVM](model/train_svm.m) and [kernel FDA](model/train_kernel_fda.m)
* (May 2019) interface added for [LIBSVM](#libsvm) and [LIBLINEAR](#liblinear)
* (Mar 2019) MVPA-Light has been integrated with FieldTrip (see [tutorial](http://www.fieldtriptoolbox.org/tutorial/mvpa_light/))
* (Feb 2019) added [kernel Fisher Discriminant Analysis](model/train_kernel_fda.m) and new metrics `precision`, `recall`, and `f1`

### Table of contents<a name="contents"></a>
1. [Installation](#installation)
2. [Overview](#overview)
3. [Classification](#classification)
4. [Regression](#regression)
5. [Examples](#examples)

## Installation <a name="installation"></a>

In Linux/Mac, open a terminal and check out the repository by typing 

```
git clone https://github.com/treder/MVPA-Light.git
```
In Windows, you might prefer to perform these steps using a [Git client](https://www.google.com/search?q=git+client+for+windows). Alternatively, you can simply download the toolbox. Git makes it easier to keep your local version up-do-date using `git pull` but it's not essential. Next, the toolbox needs to be added to Matlab's search path. In Matlab, add these lines to your `startup.m` file:

```Matlab
addpath('C:\git\MVPA-Light\startup')
startup_MVPA_Light
```

This assumes that the repository is located in `C:\git\MVPA-Light`, so change the path if necessary. The function `startup_MVPA_Light` adds the relevant folders and it avoids adding the `.git` subfolder. 

If you do not want to use the `startup.m` file, you can directly add the `MVPA-Light` folder and its subfolders to the path using [MATLAB's Path tool](https://uk.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html).

`MVPA-Light` contains two branches: the `master` branch (recommended) is the stable branch that should always work. `devel` is the development branch that contains new features that are either under construction or not tested. The toolbox has been tested with Matlab `R2012a` and newer. There may be issues with earlier Matlab versions.

## Overview <a name="overview"></a>

*Multivariate pattern analysis* (MVPA) is an umbrella term that covers many multivariate methods such classification, regression and related approaches such as Representational Similarity Analysis. `MVPA-Light` provides functions for the classification and regression of neuroimaging data. It is meant to address the basic issues in MVPA (such as classification across time and generalisation) in a fast and robust way while retaining a slim and readable codebase. For Fieldtrip users, the use of the toolbox will be familiar: The first argument to the main functions is a configuration struct `cfg` that contains all the parameters. The toolbox does *not* require or use Fieldtrip, but a FieldTrip integration is available (see [tutorial](http://www.fieldtriptoolbox.org/tutorial/mvpa_light/)).

Classifiers and regression models (jointly referred to as models) can be trained and tested by hand using the `train_*` and `test_*` functions. All classifiers and regression models are available in the [`model`](model) folder.

#### Training

In order to learn which features in the data discriminate between the experimental conditions or predict a response variable, a model needs to be exposed to *training data*. During training, the model's parameters are optimized (e.g. determining the betas in linear regression). All training functions start with `train_` (e.g. [`train_lda`](model/train_lda.m)).

#### Testing

Model performance is evaluated on a dataset called *test data*. To this end, the model is applied to samples from the test data. The predictions of the model (i.e., predicted class labels by a classifier or predicted responses by a regression model) can then be compared to the true class labels / responses in order to quantify predictive performance. All test functions start with `test_` (e.g. [`test_lda`](model/test_lda.m)).

#### Cross-validation <a name="cv"></a>

To obtain a realistic estimate of model performance and control for overfitting, a model should be tested on an independent dataset that has not been used for training. In most neuroimaging experiments, there is only one dataset with a restricted number of trials. *K-fold cross-validation* makes efficient use of this data by splitting it into k different folds. In each iteration, one of the k folds is held out and used as test set, whereas all other folds are used for training. This is repeated until every fold has been used as test set once. See [[Lemm2011]](#Lemm2011) for a discussion of cross-validation and potential pitfalls. Cross-validation is implemented in all high-level functions in MVPA-Light, i.e. [`mv_classify`](mv_classify.m), [`mv_regress`](mv_regress.m), [`mv_classify_across_time`](mv_classify_across_time.m), and [`mv_classify_timextime`](mv_classify_timextime.m). It is controlled by the following parameters:

* `cfg.cv`: cross-validation type, either 'kfold', 'leaveout', 'leavegroupout', or 'holdout' (default 'kfold')
* `cfg.k`: number of folds in k-fold cross-validation (default 5)
* `cfg.repeat`: number of times the cross-validation is repeated with new randomly assigned folds (default 5)
* `cfg.p`: if cfg.cv is 'holdout', p is the fraction of test samples (default 0.1)
* `cfg.stratify`: if 1, the class proportions are approximately preserved in each test fold (default 1)
* `cfg.group`: a vector that defines the groups if `cfg.cv = 'leavegroupout'` 

#### Hyperparameter <a name="hyperparameter"></a>

[Hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) are model parameters that have to be specified by the user. Examples are the choice of the kernel in SVM and the amount of regularization. They can be controlled by setting the `cfg.hyperparameter` field before calling any of the high-level functions. To this end, initialize the field using `cfg.hyperparameter = []`. Then, add the desired parameters, e.g. `cfg.hyperparameter.lambda = 0.5` for setting the regularisation parameter or `cfg.hyperparameter.kernel = 'polynomial'` for defining a polynomial kernel for SVM. The hyperparameters for each model are specified in the documentation for each train_ function in the folder [`model`](model/).

#### Preprocessing<a name="preprocessing"></a>

Preprocessing refers to operations applied to the data before training the classifier. In some cases, preprocessing operations such as oversampling, PCA, or Common Spatial Patterns (CSP) need to be performed as nested operations within a cross-validation analysis. In nested preprocessing, parameters are estimated on the train data and then applied to the test data. This avoids possible information flow from test set to the train set. A prepocessing pipeline can be added by setting the `cfg.preprocess` and `cfg.preprocess_param` fields. Currently implemented preprocessing functions are collected in the [`preprocess subfolder`](preprocess/). See code snippet below and [`examples/example7_preprocessing.m`](examples/example7_preprocessing.m) for examples.

#### Metrics and statistical significance

Classification and regression performance is typically measured using metrics such as accuracy and AUC for classification, or mean squared error for regression. These performance metrics do not come with p-values, however. To establish statistical significance, the function [`mv_statistics`](statistics/mv_statistics.m) can be used. It implements binomial test and permutation testing (including a cluster permutation test). See [`example9_statistics`](examples/example9_statistics.m) for code to get started.


## Classification <a name="classification"></a>

#### Introduction to classification

A *classifier* is one of the main workhorses of MVPA. The input brain data, e.g. channels or voxels, is referred to as *features*, whereas the output data is a *class label*. [Classification](https://en.wikipedia.org/wiki/Statistical_classification) is the process of taking a feature vector as input and assigning it to a class. In `MVPA-Light`, class labels must be coded as `1` (for class 1), `2` (for class 2), `3` (for class 3), and so on.

*Example*: Assume that in a ERP-based memory paradigm, the goal is to predict whether an item is remembered or forgotten based on 128-channels EEG data. The target is single-trial ERPs at t=700 ms. Then, the feature vector for each trial consists of a 128-elements vector representing the activity at 700 ms for each electrode. Class labels are "remembered" (coded as 1) and "forgotten" (coded as 2). The exact order of coding the conditions does not affect the classification performance.

#### Classifiers for two classes <a name="classifiers"></a>

* [`lda`](model/train_lda.m): Regularized Linear Discriminant Analysis (LDA). LDA searches for a projection of the data into 1D such that the class means are separated as far as possible and the within-class variability is as small as possible. To counteract overfitting, ridge regularization and shrinkage regularization are available. In shrinkage, the regularization parameter λ (lambda) ranges from λ=0 (no regularization) to λ=1 (maximum regularization). It can also be set to 'auto' to have λ be estimated automatically. For more details on regularized LDA see [[Bla2011]](#Bla2011). LDA has been shown to be formally equivalent to LCMV beamforming and it can be used for recovering time series of ERP sources [[Tre2011]](#Tre2016). See [`train_lda`](model/train_lda.m) for a full description of the parameters.

* [`logreg`](model/train_logreg.m): Logistic regression (LR). LR directly models class probabilities by fitting a logistic function to the data. Like LDA, LR is a linear classifier and hence its operation is expressed by a weight vector w and a bias b. By default, *logf* regularization is used to prevent overfitting. It is implemented by data augmentation and  does not require hyperparameters. Alternatively, L2-regularization can be used. It requires setting a positive but unbounded parameter λ (lambda) that controls the L2-penalty on the classifier weights. It can also be set to 'auto'. In this case, different λ's are tried out using a searchgrid; the value of λ maximizing cross-validation performance is then used for training on the full dataset. See [`train_logreg`](model/train_logreg.m) for a full description of the parameters.

* [`svm`](model/train_svm.m): Support Vector Machine (SVM). The parameter `c` is the cost parameter that controls the amount of regularization. It is inversely related to the λ defined above. By default, a linear SVM is used. By setting the `kernel` parameter (e.g. to 'polynomial' or 'rbf'), non-linear SVMs can be trained as well. See [`train_svm`](model/train_svm.m) for a full description of the parameters.

#### Multi-class classifiers (two or more classes)

* [`ensemble`](model/train_ensemble.m): Uses an [ensemble of classifiers](http://www.scholarpedia.org/article/Ensemble_learning) trained on random subsets of the features and random subsets of the samples. Can use any classifier with train/test functions as a learner. See [`train_ensemble`](model/train_ensemble.m) for a full description of the parameters.

* [`kernel_fda`](model/train_kernel_fda.m) : Regularized [kernel Fisher Discriminant Analysis (KFDA)](https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis). This is the kernelized version of LDA. By setting the `.kernel` parameter (e.g. to 'polynomial' or 'rbf'), non-linear classifiers can be trained. See [`train_kernel_fda`](model/train_kernel_fda.m) for a full description of the parameters.

* [`liblinear`](model/train_liblinear.m)<a name="liblinear"></a>  : interface for the [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) toolbox for linear SVM and logistic regression. It is a state-of-the-art implementation of linear SVM and Logistic Regression using compiled C code. Follow the installation and compilation instructions on the [LIBLINEAR website](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) and the [GitHub repository](https://github.com/cjlin1/liblinear). Refer to [`train_liblinear`](model/train_liblinear.m) to see how to call LIBLINEAR in `MVPA-Light`.

* [`libsvm`](model/train_libsvm.m)<a name="libsvm"></a> : interface for the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm) toolbox for Support Vector Machines (SVM). It is a state-of-the-art implementation of SVM using compiled C code. Follow the installation and compilation instructions on the [LIBSVM website](https://www.csie.ntu.edu.tw/~cjlin/libsvm) and the [GitHub repository](https://github.com/cjlin1/libsvm). Refer to [`train_libsvm`](model/train_libsvm.m) to see how to call LIBSVM in `MVPA-Light`. (*note*: LIBSVM provides a function called `svmtrain` which is overshadowed by a Matlab function of the same name in some Matlab versions. This can lead to execution errors. Type `which -all svmtrain` to check whether there is indeed multiple versions.)

* [`multiclass_lda`](model/train_multiclass_lda.m) : Regularized multi-class Linear Discriminant Analysis (LDA). The data is first projected onto a (C-1)-dimensional discriminative subspace, where C is the number of classes. A new sample is assigned to the class with the closest centroid in this subspace. See [`train_multiclass_lda`](model/train_multiclass_lda.m) for a full description of the parameters.

* [`naive_bayes`](model/train_naive_bayes.m)<a name="naivebayes"></a> : Gaussian Naive Bayes classifier. Its naive assumption is that, given the class label, the features are independent of each other. This allows the posterior probability to be expressed in terms of univariate densities. At testing time, the maximum a posteriori (MAP) rule is applied to assign a sample to the class with the maximum posterior probability. See [`train_naive_bayes`](model/train_naive_bayes.m) for a full description of the parameters.

#### Classification across time
Many neuroimaging datasets have a 3-D structure (trials x channels x time). The start of the trial (t=0) typically corresponds to stimulus or response onset. Classification across time can help identify at which time point in a trial discriminative information shows up. To this end, classification is performed across trials, for each time point separately. This is implemented in the function [`mv_classify_across_time`](mv_classify_across_time.m). It returns classification performance calculated for each time point in a trial. [`mv_plot_result`](plot/mv_plot_result.m) can be used to plot the result.

#### Time x time generalization

Classification across time does not give insight into whether information is shared across different time points. For example, is the information that the classifier uses early in a trial (t=80 ms) the same that it uses later (t=300ms)? In time generalization, this question is answered by training the classifier at a certain time point t. The classifer is then tested at the same time point t but it is also tested at all *other* time points in the trial [[King2014]](#King2014). [`mv_classify_timextime`](mv_classify_timextime.m) implements time generalization. It returns a 2D matrix of classification performance, with performance calculated for each combination of training time point and testing time point. [`mv_plot_result`](plot/mv_plot_result.m) can be used to plot the result.

#### Classification of multi-dimensional data<a name="mvclassify"></a>

Neuroimaging datasets can be high dimensional. For instance, time-frequency data can have 4 (e.g. samples x channels x frequencies x times) or more dimensions. The function [`mv_classify`](mv_classify.m) deals with data of an arbitrary number and order of dimensions. It combines and generalizes the capabilities of the other high-level functions and allows for flexible tailoring of classification analysis including frequency x frequency generalization.

[`mv_classify`](mv_classify.m) also implements searchlight analysis, which investigates which features contribute most to classification performance. The answer to this question can be used to better interpret the data or to perform feature selection. If there is a spatial structure in the features (e.g. neighbouring eletrodes, neighbouring voxels), groups of features rather than single features can be considered. The result is a classification performance measure for each feature. If the features are e.g. channels, the result can be plotted as a topography. See [`example6_classify_multidimensional_data.m`](examples/example6_classify_multidimensional_data.m) for code to get you started.

#### Classification performance metrics <a name="metrics"></a>

Classifier output comes in form of decision values (=distances to the hyperplane for linear methods) or directly in form of class labels. However, rather than looking at the raw classifier output, one is often only interested in a performance metric that summarizes how well the classifier discriminates between the classes. The following metrics can be calculated by the function [`mv_calculate_performance`](utils/mv_calculate_performance.m):

* `accuracy` (can be abbreviated as `acc`): Classification accuracy, representing the fraction correctly predicted class labels.
* `auc`: Area under the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). An alternative to classification accuracy that is more robust to imbalanced classes and independent of changes to the classifier threshold.
* `confusion`: [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). Rows corresponds to true class labels, columns correspond to predicted class labels. The (i,j)-th element gives the proportion of samples of class i that have been classified as class j.
* `dval`: Average decision value for each class.
* `f1`: [F1 score](https://en.wikipedia.org/wiki/F1_score) is the harmonic average of precision and recall, given by `2 *(precision * recall) / (precision + recall)`.
* `kappa`: <a name="kappa"></a> [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) measures the 'inter-rater reliability' between predicted and actual class labels. See here for a [discussion on CrossValidated](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english).
* `precision`: [precision](https://en.wikipedia.org/wiki/Precision_and_recall) is given as the number of true positives divided by true positives plus false positives. For multi-class, it is calculated per class from the confusion matrix by dividing each diagonal element by the row sum. 
* `recall`: [recall](https://en.wikipedia.org/wiki/Precision_and_recall) is given as the number of true positives divided by true positives plus false negatives. For multi-class, it is calculated per class from the confusion matrix by dividing each diagonal element by the respective column sum. 
* `tval`: for two classes, calculates the [t-test statistic](https://en.wikipedia.org/wiki/Student's_t-test#Equal_or_unequal_sample_sizes.2C_equal_variance) for unequal sample size, equal variance case, based on the decision values. Can be useful for a subsequent second-level analysis across subjects.
* `none`: Does not calculate a metric but returns the raw classifier output instead. 

There is usually no need to call [`mv_calculate_performance`](utils/mv_calculate_performance.m) directly. By setting the `cfg.metric` field, the performance metric is calculated automatically in [`mv_classify_across_time`](mv_classify_across_time.m), [`mv_classify_timextime`](mv_classify_timextime.m) and [`mv_classify`](mv_classify.m). You can provide a cell array of metrics, e.g. `cfg.metric = {'accuracy', 'confusion'}` to calculate multiple metrics at once.


## Regression <a name="regression"></a>

#### Introduction to regression

A *regression model* performs statistical predictions, similar to a classifier. The main difference between both types of models is that a classifier predicts class labels (a discrete variable) whereas a regression model predicts responses (a continuous variable). 

*Example*: We hypothesize that reaction time (RT) in a task is predicted by the magnitude of the BOLD response in brain areas. To investigate this, we perform a searchlight analysis: In every iteration, the single-trial BOLD response in a subset of contiguous voxels serves as features, whereas RT serves as response variable. We assume that this relationship is non-linear, so we use [kernel ridge regression](model/train_kernel_ridge.m) to predict RT.

#### Regression models <a name="regressionmodels"></a>

* [`ridge`](model/train_ridge.m)<a name="ridge"></a>: Ridge regression (includes linear regression). To counteract overfitting, the hyperparameter λ (lambda) controls the amount of regularization. For λ=0 (no regularization), ridge regression is equal to linear regression. Setting λ>0 is necessary in cases of [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity), which is common in neuroimaging data due to high correlations between neighbouring channels/voxels. See [`train_ridge`](model/train_ridge.m) for a full description of the parameters.
* [`kernel_ridge`](model/train_kernel_ridge.m)<a name="kernelridge"></a>: [Kernel Ridge regression](https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf) is the kernelized version of ridge regression. It allows the modelling of non-linear relationships by setting the `.kernel` hyperparameter (e.g. to 'polynomial' or 'rbf').  See [`train_kernel_ridge`](model/train_kernel_ridge.m) for a full description of the parameters.
* [LIBSVM](#libsvm) and [LIBLINEAR](#liblinear): both packages provide Support Vector Regression (SVR) models in addition to classifiers. For [LIBSVM](#libsvm), setting the hyperparameter `svm_type` allows for the training of epsilon-SVR (svm_type = 3) and nu-SVR (svm_type = 4). [LIBLINEAR](#liblinear) allows for the training of *linear* SVR by setting the hyperparameter `type` allows, namely primal L2-loss SVR (type = 11), dual L2-loss SVR (type = 12), and L1-loss SVR (type = 13). See [`train_libsvm`](model/train_libsvm.m) and [`train_liblinear`](model/train_liblinear.m) for a full description of the parameters.

#### Regression performance metrics

Often, one is not interested in the concrete predictions of the regression model, but rather in a performance metric that quantifies how good the predictions are. The following metrics can be calculated by the function [`mv_calculate_performance`](utils/mv_calculate_performance.m):

* [`mean_absolute_error`](https://en.wikipedia.org/wiki/Mean_absolute_error) (can be abbreviated as `mae`): average absolute prediction error. The absolute prediction error for a given sample is |y - ŷ|, where ŷ is the model prediction and y is the true response.
* [`mean_squared_error`](https://en.wikipedia.org/wiki/Mean_squared_error) (can be abbreviated as `mse`): average squared prediction error. The squared prediction error for a given sample is (y - ŷ)^2.
* [`r_squared`](https://en.wikipedia.org/wiki/Coefficient_of_determination): proportion of variance in the response variable that is predicted by the model.
* `none`: Does not calculate a metric but returns the predictions instead. 

There is usually no need to call [`mv_calculate_performance`](utils/mv_calculate_performance.m) directly. By setting the `cfg.metric` field, the performance metric is calculated automatically in [`mv_regress`](mv_regress.m). You can provide a cell array of metrics, e.g. `cfg.metric = {'mae', 'mse'}` to calculate multiple metrics at once.


## Examples <a name="examples"></a>

This section gives some basic examples. More detailed examples and data can be found in the [`examples/`](examples) subfolder.

#### Training and testing a classifier by hand

```Matlab

% Load data (in /examples folder)
[dat,clabel] = load_example_data('epoched3');

% Fetch the data from the 100th time sample
X = dat.trial(:,:,100);

% Get default hyperparameters for the classifier
param = mv_get_hyperparameter('lda');

% Train an LDA classifier
cf = train_lda(param, X, clabel);

% Test classifier on the same data and get the predicted labels
predlabel = test_lda(cf, X);

% Calculate classification accuracy
acc = mv_calculate_performance('accuracy','clabel',predlabel,clabel)
```

See [`examples/example1_train_and_test.m`](examples/example1_train_and_test.m) for more details. In most cases, you would not perform training/testing by hand but rather call one of the high-level functions described below.

<!-- 
#### Cross-validation


```Matlab
cfg = [];
cfg.classifier      = 'lda';
cfg.metric          = 'accuracy';
cfg.cv              = 'kfold';
cfg.k               = 5;
cfg.repeat          = 2;

% Perform 5-fold cross-validation with 2 repetitions.
% As classification performance measure we request accuracy (acc).
acc = mv_crossvalidate(cfg, X, clabel);
```

See [`examples/example2_crossvalidate.m`](examples/example2_crossvalidate.m) for more details.

-->

#### Classification across time

```Matlab

% Classify across time using default settings
cfg = [];
acc = mv_classify_across_time(cfg, dat.trial, clabel);

```

See [`examples/example3_classify_across_time.m`](examples/example3_classify_across_time.m) for more details.

#### Time generalisation (time x time classification)


```Matlab
cfg = [];
cfg.metric     = 'auc';

auc = mv_classify_timextime(cfg, dat.trial, clabel);

```

See [`examples/example4_classify_timextime.m`](examples/example4_classify_timextime.m) for more details.

<!-- 
#### Searchlight analysis

```Matlab
cfg = [];
acc = mv_searchlight(cfg, dat.trial, clabel);

```

See [`examples/example5_searchlight.m`](examples/example5_searchlight.m) for more details.
-->

#### Preprocessing pipeline

```Matlab
cfg =  [];
cfg.preprocess = {'pca' 'average_samples'};
acc = mv_classify_across_time(cfg, dat.trial, clabel);

```

See [`examples/example7_preprocessing.m`](examples/example7_preprocessing.m) for more details.

#### Ridge Regression

```Matlab
cfg                = [];
cfg.model          = 'ridge';
cfg.metric         = 'mse';
cfg.hyperparameter = [];
cfg.hyperparameter.lambda = 0.1;

mse = mv_regress(cfg, X, y);
```

See [`examples/example8_regression.m`](examples/example8_regression.m) for more details.


### References

[Bla2011<a name="Bla2011">]  [Blankertz, B., Lemm, S., Treder, M., Haufe, S., & Müller, K. R. (2011). Single-trial analysis and classification of ERP components - A tutorial. NeuroImage, 56(2), 814–825.](http://www.sciencedirect.com/science/article/pii/S1053811910009067)

[King2014<a name="King2014">] [King, J.-R., & Dehaene, S. (2014). Characterizing the dynamics of mental representations: the temporal generalization method. Trends in Cognitive Sciences, 18(4), 203–210.](https://doi.org/10.1016/j.tics.2014.01.002)

[Lemm2011<a name="Lemm2011">]  [Lemm, S., Blankertz, B., Dickhaus, T., & Müller, K. R. (2011). Introduction to machine learning for brain imaging. NeuroImage, 56(2), 387–399.](http://www.sciencedirect.com/science/article/pii/S1053811910014163)

[Tre2016<a name="Tre2016">]  [Treder, M. S., Porbadnigk, A. K., Shahbazi Avarvand, F., Müller, K.-R., & Blankertz, B. (2016). The LDA beamformer: Optimal estimation of ERP source time series using linear discriminant analysis. NeuroImage, 129, 279–291.](https://doi.org/10.1016/j.neuroimage.2016.01.019)
