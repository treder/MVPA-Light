# MVPA-Light
Lightweight Matlab toolbox for multivariate pattern analysis (MVPA)

## Installation
Download the toolbox or, better, clone it using Git. In Matlab, you can add the following line to your `startup.m` file to add the MVPA-Light toolbox to the Matlab path:

```Matlab
addpath(genpath('my-path-to/MVPA-Light/'))
```

The Git repository is split into two branches: the `master` branch (recommended) is the stable branch that should always work. The `devel` branch (not recommended) is the development branch that contains new features that are either under construction or not tested.

## Overview
`MVPA-Light` provides functions for the multivariate classification of ERPs/ERFs and oscillations. For Fieldtrip users, the use of the toolbox will be familiar: The first argument to the main functions is a configuration struct `cfg` that contains all the parameters. However, the toolbox does *not* require or use Fieldtrip. `MVPA-Light` is intended to remain small with a readable and well-documented codebase.

Classifiers can be trained and tested directly using the train_* and test_* functions. For data with a trial structure, `mv_classify_across_time` can be used to obtain classification performance for each time point in a trial. `mv_classify_timextime` implements time generalisation, i.e., training on a specific time point, and testing the classifier on all other time points in a trial. Cross-validation, balancing unequal class proportions, and different performance metrics are automatically implemented in these functions.

## Classifiers

### take from wiki page and then extend here


## testing and training 

A classifier is the main workhorse of MVPA. Its task is to take input data such as EEG activity, called 'features' in machine learning, and predict the class label ... 


however, in order to learn ... 
The process of tuning a classifier to do its job by exposing it to data is knowed as 'training'. During training, the classifier learns which features are important and which are not.

For more details (on LDA), see [1] --- Blankertz et al

Linear Discriminant Analysis (LDA) (equivalent to Fisher's Discriminant Analysis for two classes), linear support vector machines (SVM), and logistic regression, are all linear classifiers that perform classification by means of a linear hyperplane. The only ...
As a rule of thumb, ... 

## Cross-validation/Resampling methods

a complex classifier (such as kernel SVM) can easily learn to perfectly discriminate a specific dataset. However, the same classifier will perform very poorly when applied to a new dataset, called D2. The reason is that the classifier has not learnt ... Instead, it has overadapted to the 

complex methods tend to overfit the data ... 

...
To get a realistic estimate of the classifier performance, a classifier should be applied to a new dataset that has not been used for learning.

## Classification across time
In many EEG/MEG experiments, data is split into epochs representing individual trials. 
 has a trial structure ... 

* Classification across time (mv_classify_across_time): 

* Time x time generalisation (mv_classify_timextime)

* Time x time generalisation using two datasets (mv_classify_timextime_two_datasets): 

## Time x time generalisation






