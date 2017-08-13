# MVPA-Light
Light-weight Matlab toolbox for multivariate pattern analysis (MVPA)

### Table of contents
1. [Installation](#installation)
2. [Overview](#overview)
3. [Classification](#classification)
4. [Cross-validation](#crossvalidation)

## Installation <a name="installation"></a>
Download the toolbox or, better, clone it using Git. In Matlab, you can add the following line to your `startup.m` file to add the MVPA-Light toolbox to the Matlab path:

```Matlab
addpath(genpath('my-path-to/MVPA-Light/'))
```

The Git repository is split into two branches: the `master` branch (recommended) is the stable branch that should always work. `devel` is the development branch that contains new features that are either under construction or not tested.

## Overview <a name="overview"></a>
`MVPA-Light` provides functions for the binary classification of neuroimaging data. For Fieldtrip users, the use of the toolbox will be familiar: The first argument to the main functions is a configuration struct `cfg` that contains all the parameters. However, the toolbox does *not* require or use Fieldtrip. `MVPA-Light` is intended to remain small with a readable and well-documented codebase.

Classifiers can be trained and tested directly using the `train_*` and `test_*` functions. For data with a trial structure, such as ERP datasets, `mv_classify_across_time` can be used to obtain classification performance for each time point in a trial. `mv_classify_timextime` implements time generalisation, i.e., training on a specific time point, and testing the classifier on all other time points in a trial. `mv_classify_timextime_two_datasets` is used when training is performed on one dataset and testing is performed on a second dataset.
Cross-validation, balancing unequal class proportions, and different performance metrics are automatically implemented in these functions.

## Classification <a name="classification"></a>

#### Introduction

<!---In cognitive neuroscience, the term *decoding* refers to the prediction of experimental conditions or mental states (output) based on multivariate brain data (input). The term *classification* means the same. Note that classification is the standard term in machine learning and many other disciplines whereas decoding is specific to cognitive neuroscience. *Multivariate pattern analysis* (MVPA) is an umbrella term that covers many multivariate methods such classification and related approaches such as Representational Similarity Analysis (RSA). --->

A *classifier* is the main workhorse of MVPA. The input brain data, e.g. channels or voxels, is referred to as *features*, whereas the output data is a *class label*. The classifier takes a feature vector as input and assigns it to a class. For binary classification, class labels are often coded as +1 (for class 1) and -1 (for class 2).

<!-- *Example*: Assume that in a ERP-based memory paradigm, the goal is to predict whether an item is remembered or forgotten based on 128-channels EEG data. The target is single-trial ERPs at t=700 ms. Then, the feature vector for each trial consists of a 128-elements vector representing the activity at 700 ms for each electrode. Class labels are "remembered" (coded as +1) and "forgotten" (coded as -1). Note that the exact coding does not affect the classification.
-->

#### Training

In order to learn which features in the data discriminate between the experimental conditions, a classifier needs to be exposed to *training data*. During training, the classifier's parameters are optimised (analogous to determining the beta's in linear regression). All training functions start with `train_` (e.g. `train_lda`).

#### Testing

Classifier performance is evaluated on a dataset called *test data*. To this end, the classifier is applied to samples from the test data. The class label predicted by the classifier can then be compared to the true class label in order to quantify classification performance. All test functions start with `test_` (e.g. `test_lda`).

#### Classifiers

* `train_lda`, `test_lda`: Regularised Linear Discriminant Analysis (LDA). For two classes, LDA is equivalent to Fisher's discriminant analysis (FDA). Hence, LDA searches for a projection of the data into 1D such that the class means are separated as far as possible and the within-class variability is as small as possible. To prevent overfitting and assure invertibility of the covariance matrix, the regularisation parameter λ can be varied between λ =0 (no regularisation) and λ=1 (maximum regularisation). It can also be set to λ='auto'. In this case, λ is estimated automatically. For more details on regularised LDA see [[Bla2011]](#Bla2011).

<!-- * `train_svm`, `test_svm`: Support vector machines (SVM). Uses the [LIBSVM package](https://github.com/arnaudsj/libsvm) that needs to be installed.
* `train_logist`, `test_logist`: Logistic regression classifier using Lucas Parra's implementation.
-->

#### Classification across time
Many neuroimaging datasets have a 3-D structure (trials x channels x time). The start of the trial (t=0) typically corresponds to stimulus or response onset. Classification across time can help identify at which time point in a trial discriminative information shows up. To this end, classification is performed across trials, for each time point separately. This is implemented in the function `mv_classify_across_time`. It returns classification performance calculated for each time point in a trial. `mv_plot_1D` can be used to plot the result.


#### Time x time generalisation

Classification across time does not give insight into whether information is shared across different time points. For example, is the information that the classifier uses early in a trial (t=80 ms) the same that it uses later (t=300ms)? In time generalisation, this question is answered by training the classifier at a certain time point t. The classifer is then tested at the same time point t but it is also tested at all *other* time points in the trial. `mv_classify_timextime` implements time generalisation. It returns a 2D matrix of classification performance, with performance calculated for each combination of training time point and testing time point. If two separate datasets are used, one for training and one for testing, the function `mv_classify_timextime_two_datasets` can be used instead.
`mv_plot_2D` can be used to plot the result.

<!--
## Q&A

#### Which classifier should I use?

Note that all linear classifiers (LDA, Logistic regression, linear SVM) try to find a hyperplane that optimally separates the two classes. They only differ in the way ...

As a rule of thumb,

#### Which classifier performance measure should I use?
-->

### References

[Bla2011<a name="Bla2011">]  [Blankertz, B., Lemm, S., Treder, M., Haufe, S., & Müller, K. R. (2011). Single-trial analysis and classification of ERP components - A tutorial. NeuroImage, 56(2), 814–825.](http://www.sciencedirect.com/science/article/pii/S1053811910009067)

[Lemm2011<a name="Lemm2011">]  [Lemm, S., Blankertz, B., Dickhaus, T., & Müller, K. R. (2011). Introduction to machine learning for brain imaging. NeuroImage, 56(2), 387–399.](http://www.sciencedirect.com/science/article/pii/S1053811910014163)

[Treder2016<a name="Treder2016">]  [Treder, M. S., Porbadnigk, A. K., Shahbazi Avarvand, F., Müller, K.-R., & Blankertz, B. (2016). The LDA beamformer: Optimal estimation of ERP source time series using linear discriminant analysis. NeuroImage, 129, 279–291.](https://doi.org/10.1016/j.neuroimage.2016.01.019)
