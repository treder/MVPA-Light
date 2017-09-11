# MVPA-Light
Light-weight Matlab toolbox for multivariate pattern analysis (MVPA)

### Table of contents
1. [Installation](#installation)
2. [Overview](#overview)
3. [Classification](#classification)
4. [Examples](#examples)

## Installation <a name="installation"></a>
Download the toolbox or, better, clone it using Git. In Matlab, you can add the following line to your `startup.m` file to add the MVPA-Light toolbox to the Matlab path:

```Matlab
addpath(genpath('my-path-to/MVPA-Light/'))
```

The Git repository contains two branches: the `master` branch (recommended) is the stable branch that should always work. `devel` is the development branch that contains new features that are either under construction or not tested.

## Overview <a name="overview"></a>
`MVPA-Light` provides functions for the binary classification of neuroimaging data. It is meant to address the basic issues in MVPA (such as classification across time and generalisation) in a fast and robust way while retaining a slim and readable codebase. For Fieldtrip users, the use of the toolbox will be familiar: The first argument to the main functions is a configuration struct `cfg` that contains all the parameters. However, the toolbox does *not* require or use Fieldtrip.

Classifiers can be trained and tested by hand using the `train_*` and `test_*` functions. For data with a trial structure, such as ERP datasets, `mv_classify_across_time` can be used to obtain classification performance for each time point in a trial. `mv_classify_timextime` implements time generalisation, i.e., training on a specific time point, and testing the classifier on all other time points in a trial. Cross-validation, balancing unequal class proportions, and different performance metrics are automatically implemented in these functions.

## Classification <a name="classification"></a>

#### Introduction

<!---In cognitive neuroscience, the term *decoding* refers to the prediction of experimental conditions or mental states (output) based on multivariate brain data (input). The term *classification* means the same. Note that classification is the standard term in machine learning and many other disciplines whereas decoding is specific to cognitive neuroscience. *Multivariate pattern analysis* (MVPA) is an umbrella term that covers many multivariate methods such classification and related approaches such as Representational Similarity Analysis (RSA). --->

A *classifier* is the main workhorse of MVPA. The input brain data, e.g. channels or voxels, is referred to as *features*, whereas the output data is a *class label*. The classifier takes a feature vector as input and assigns it to a class. In `MVPA-Light`, class labels must be coded as `1` (for class 1) and `2` (for class 2).

<!-- *Example*: Assume that in a ERP-based memory paradigm, the goal is to predict whether an item is remembered or forgotten based on 128-channels EEG data. The target is single-trial ERPs at t=700 ms. Then, the feature vector for each trial consists of a 128-elements vector representing the activity at 700 ms for each electrode. Class labels are "remembered" (coded as +1) and "forgotten" (coded as -1). Note that the exact coding does not affect the classification.
-->

#### Training

In order to learn which features in the data discriminate between the experimental conditions, a classifier needs to be exposed to *training data*. During training, the classifier's parameters are optimised (analogous to determining the beta's in linear regression). All training functions start with `train_` (e.g. `train_lda`).

#### Testing

Classifier performance is evaluated on a dataset called *test data*. To this end, the classifier is applied to samples from the test data. The class label predicted by the classifier can then be compared to the true class label in order to quantify classification performance. All test functions start with `test_` (e.g. `test_lda`).

#### Classifiers

* `train_lda`, `test_lda`: Regularised Linear Discriminant Analysis (LDA). For two classes, LDA is equivalent to Fisher's discriminant analysis (FDA). Hence, LDA searches for a projection of the data into 1D such that the class means are separated as far as possible and the within-class variability is as small as possible. To prevent overfitting and assure invertibility of the covariance matrix, the regularisation parameter λ can be varied between λ=0 (no regularisation) and λ=1 (maximum regularisation). It can also be set to λ='auto'. In this case, λ is estimated automatically. For more details on regularised LDA see [[Bla2011]](#Bla2011). LDA has been shown to be formally equivalent to LCMV beamforming and it can be used for recovering time series of ERP sources [[Tre2011]](#Tre2011).
* `train_ensemble`, `test_ensemble`: Uses an ensemble of classifiers trained on random subsets of the features and random subsets of the samples. Can use any classifier with train/test functions as a learner.

<!--
* `train_svm`, `test_svm`: Support vector machines (SVM). Uses the [LIBSVM package](https://github.com/arnaudsj/libsvm) that needs to be installed.
* `train_logist`, `test_logist`: Logistic regression classifier using Lucas Parra's implementation. See `external/logist.m` for an explanation of the hyperparameters.
-->

#### Classification across time
Many neuroimaging datasets have a 3-D structure (trials x channels x time). The start of the trial (t=0) typically corresponds to stimulus or response onset. Classification across time can help identify at which time point in a trial discriminative information shows up. To this end, classification is performed across trials, for each time point separately. This is implemented in the function `mv_classify_across_time`. It returns classification performance calculated for each time point in a trial. `mv_plot_1D` can be used to plot the result.


#### Time x time generalisation

Classification across time does not give insight into whether information is shared across different time points. For example, is the information that the classifier uses early in a trial (t=80 ms) the same that it uses later (t=300ms)? In time generalisation, this question is answered by training the classifier at a certain time point t. The classifer is then tested at the same time point t but it is also tested at all *other* time points in the trial [[King2014]](#King2014). `mv_classify_timextime` implements time generalisation. It returns a 2D matrix of classification performance, with performance calculated for each combination of training time point and testing time point. `mv_plot_2D` can be used to plot the result.


#### Classifier performance metrics

Classifier output comes in form of decision values (=distances to the hyperplane for linear methods) or directly in form of class labels. However,  one is often only interested in a performance metric that summarises how well the classifier discriminates between the classes. The following metrics can be calculated by the function `mv_classifier_performance`:

* `acc`: Classification accuracy, representing the fraction correctly predicted class labels.
* `auc`: Area under the ROC curve. An alternative to classification accuracy that is more robust to imbalanced classes and independent of changes to the classifier threshold.
* `dval`: Average decision value for each class.

Performance metrics can be selected in `mv_classify_across_time` and `mv_classify_timextime` by setting the `cfg.metric` field.


#### Cross-validation

To obtain a realistic estimate of classifier performance and control for overfitting, a classifier should be tested on an independent dataset that has not been used for training. In most neuroimaging experiments, there is only one dataset with a restricted number of trials. *K-fold cross-validation* makes efficient use of this data by splitting it into k different folds. In each iteration, one of the k folds is held out and used as test set, whereas all other folds are used for training. This is repeated until every fold has been used as test set once. See [[Lemm2011]](#Lemm2011) for a discussion of cross-validation and potential pitfalls. Cross-validation is controlled by the parameters `cfg.CV`, `cfg.K`, and `cfg.repeat`.


## Examples<a name="examples"></a>

This section gives some basic examples. More detailed examples and data can be found in the `examples/` subfolder.

#### Training and testing by hand

```Matlab
% Load example data
load('epoched1')

% Determine the class labels
clabel = zeros(nTrial, 1);
clabel(attended_deviant)  = 1;
clabel(~attended_deviant) = 2;

% Fetch the data from the 100th time sample
X = dat.trial(:,:,100);

% Get default hyperparameters for the classifier
cfg_lda = mv_classifier_defaults('lda');

% Train an LDA classifier
cf = train_lda(cfg_lda, X, clabel);

% Test classifier on the same data and get the predicted labels
predlabel = test_lda(cf, X);

% Calculate classification accuracy
acc = mv_classifier_performance('acc',predlabel,clabel)

```

See `examples/example1_train_and_test.m` for more details.

#### Cross-validation


```Matlab
ccfg = [];
ccfg.classifier      = 'lda';
ccfg.param           = struct('lambda','auto');
ccfg.metric          = 'acc';
ccfg.CV              = 'kfold';
ccfg.K               = 5;
ccfg.repeat          = 3;
ccfg.balance         = 'undersample';
ccfg.verbose         = 1;

acc = mv_crossvalidate(ccfg, X, clabel);
```

See `examples/example2_crossvalidate.m` for more details.


#### Classification across time


```Matlab
ccfg =  [];
ccfg.CV         = 'kfold';
ccfg.K          = 5;
ccfg.repeat     = 5; % 10
ccfg.classifier = 'lda';
ccfg.param      = struct('lambda','auto');
ccfg.verbose    = 1;

acc = mv_classify_across_time(ccfg, dat.trial, clabel);

```

See `examples/example3_classify_across_time.m` for more details.

#### Time generalisation (time x time classification)


```Matlab
ccfg =  [];
ccfg.classifier = 'lda';
ccfg.param      = struct('lambda','auto');
ccfg.verbose    = 1;
ccfg.normalise  = 'demean';
ccfg.metric     = {'acc' 'auc'};

[acc,auc] = mv_classify_timextime(ccfg, dat.trial, clabel);

```

See `examples/example4_classify_timextime.m` for more details.




<!--
## Q&A

#### Which classifier should I use?

Note that all linear classifiers (LDA, Logistic regression, linear SVM) try to find a hyperplane that optimally separates the two classes. They only differ in the way ...

As a rule of thumb,

#### Which classifier performance measure should I use?
-->

### References

[Bla2011<a name="Bla2011">]  [Blankertz, B., Lemm, S., Treder, M., Haufe, S., & Müller, K. R. (2011). Single-trial analysis and classification of ERP components - A tutorial. NeuroImage, 56(2), 814–825.](http://www.sciencedirect.com/science/article/pii/S1053811910009067)

[King2014<a name="King2014">] [King, J.-R., & Dehaene, S. (2014). Characterizing the dynamics of mental representations: the temporal generalization method. Trends in Cognitive Sciences, 18(4), 203–210.](https://doi.org/10.1016/j.tics.2014.01.002)

[Lemm2011<a name="Lemm2011">]  [Lemm, S., Blankertz, B., Dickhaus, T., & Müller, K. R. (2011). Introduction to machine learning for brain imaging. NeuroImage, 56(2), 387–399.](http://www.sciencedirect.com/science/article/pii/S1053811910014163)

[Tre2016<a name="Tre2016">]  [Treder, M. S., Porbadnigk, A. K., Shahbazi Avarvand, F., Müller, K.-R., & Blankertz, B. (2016). The LDA beamformer: Optimal estimation of ERP source time series using linear discriminant analysis. NeuroImage, 129, 279–291.](https://doi.org/10.1016/j.neuroimage.2016.01.019)
