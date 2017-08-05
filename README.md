# MVPA-Light
Lightweight Matlab toolbox for multivariate pattern analysis (MVPA)

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

Classifiers can be trained and tested directly using the train_* and test_* functions. For data with a trial structure, such as ERP datasets, `mv_classify_across_time` can be used to obtain classification performance for each time point in a trial. `mv_classify_timextime` implements time generalisation, i.e., training on a specific time point, and testing the classifier on all other time points in a trial. Cross-validation, balancing unequal class proportions, and different performance metrics are automatically implemented in these functions.

## Classification <a name="classification"></a>

### Introduction

<!---In cognitive neuroscience, the term *decoding* refers to the prediction of experimental conditions or mental states (output) based on multivariate brain data (input). The term *classification* means the same. Note that classification is the standard term in machine learning and many other disciplines whereas decoding is specific to cognitive neuroscience. *Multivariate pattern analysis* (MVPA) is an umbrella term that covers many multivariate methods such classification and related approaches such as Representational Similarity Analysis (RSA). --->

A *classifier* is the main workhorse of MVPA. The input brain data, e.g. channels or voxels, is referred to as *features*, whereas the output data is a *class label*. The classifier takes a feature vector as input and assigns it to a class. For binary classification, class labels are often coded as +1 (for class 1) and -1 (for class 2).

*Example*: Assume that in a ERP-based memory paradigm, the goal is to predict whether an item is remembered or forgotten based on 128-channels EEG data. The target is single-trial ERPs at t=700 ms. Then, the feature vector for each trial consists of a 128-elements vector representing the activity at 700 ms for each electrode. Class labels are "remembered" (coded as +1) and "forgotten" (coded as -1). Note that the exact coding does not affect the classification.

### Training

In order to learn which features in the data discriminate between the experimental conditions, a classifier needs to be exposed to data called  *training data*. During training, the classifier's parameters are optimised (analogous to determining the beta's in linear regression).

### Testing

After training, classifier performance is evaluated on a dataset called *test data*. To this end, the classifier is applied to samples from the test data. The class label predicted by the classifier can then be compared to the true class label in order to quantify classification performance.



### Classifiers

* Linear Discriminant Analysis (LDA) (`train_lda`, `test_lda`): For two classes, LDA is equivalent to Fisher's Discriminant Analysis. LDA models ... For more details on regularised LDA, see [1] --- Blankertz et al
* linear support vector machines (SVM)
* logistic regression, are all linear classifiers that perform classification by means of a linear hyperplane. The only ...

## Cross-validation <a name="crossvalidation"></a>

a complex classifier (such as kernel SVM) can easily learn to perfectly discriminate a specific dataset. However, the same classifier will perform very poorly when applied to a new dataset, called D2. The reason is that the classifier has not learnt ... Instead, it has overadapted to the

complex methods tend to overfit the data ...

...
To get a realistic estimate of the classifier performance, a classifier should be applied to a new dataset that has not been used for learning.

## Classification across time
In many EEG/MEG experiments, data is split into epochs representing individual trials.
 has a trial structure ...

* Classification across time (`mv_classify_across_time`):

* Time x time generalisation (`mv_classify_timextime`)

* Time x time generalisation using two datasets (`mv_classify_timextime_two_datasets`):

## Time x time generalisation
