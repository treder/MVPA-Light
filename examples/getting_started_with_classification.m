% GETTING STARTED WITH CLASSIFICATION
% 
% This is the go-to tutorial if you are new to the toolbox and want to
% get started with classifying data. It covers the following topics:
%
% (1) Loading example data
% (2) Cross-validation and explanation of the cfg struct
% (3) Classification of data with a time dimension
% (4) Time generalization (time x time classification)
% (5) Plotting results
% (6) Searchlight analysis
% (7) Hyperparameters
%
% It is recommended that you work through this tutorial step by step. To
% this end, copy the lines of code that you are currently reading and paste
% them into the Matlab console. 
%
% There will be exercises throughout the tutorial. Try to do the exercise,
% you can then check your code against the solution at the end of this
% file.
%
% Note: we will use machine learning terminology such as cross-validation,
% features, classifier, train and test set. If you are unfamiliar with
% these terms, please check out the glossary in section 1.1 of the
% MVPA-Light paper (www.frontiersin.org/articles/10.3389/fnins.2020.00289)
% and read the tutorial papers that are mentioned there.
%
% Troubleshooting: If the code crashes on your computer, make sure that you
% always have the latest version of the toolbox and that you are using
% Matlab version 2012 or newer.
%
% Documentation:
% The Github Readme file is the most up-to-date documentation of the
% toolbox. You will find an explanation of the functions, models,
% metrics and parameters there: github.com/treder/MVPA-Light/blob/master/README.md
%
% Next steps: Once you finished this tutorial, you can continue with one
% of the other tutorials:
% - advanced_classification
% - understanding_metrics
% - understanding_preprocessing
% - understanding_statistics
% - understanding_train_and_test_functions
%
% You can also check out the Github repo https://github.com/treder/MVPA-Light-Paper
% It contains all the analysis scripts used in the MVPA-Light paper.

close all
clear

%% (1) Loading example data
% MVPA-Light comes with example datasets to get you started right away. 
% The dataset is taken from this study: 
% https://iopscience.iop.org/article/10.1088/1741-2560/11/2/026009/meta
% The data has been published at
% http://bnci-horizon-2020.eu/database/data-sets (dataset #15). 
% A subset of this dataset has been shipped with MVPA-Light so you do not
% need to download the data.
%
% Here, it will be explained how to load the data and how to use it.
% MVPA-Light has a custom function, load_example_data, which is exclusively
% used for loading the example datasets shipped with the toolbox. Let's
% load one of the datasets:

% Load data (which is located in the MVPA-Light/examples folder)
% (copy the following line and paste it into the Matlab console)
[dat, clabel] = load_example_data('epoched3');

% We loaded a dataset called 'epoched3' and it returned two variables, dat
% and clabel. Now type dat in the console:
dat

% dat is actually a FieldTrip structure, but we are only interested in the
% data contained in the dat.trial field for now. Let us assign this data to 
% the variable X and then look at the size of X:
% (copy the following two lines and paste them into the Matlab console)
X = dat.trial;
size(X)

% It has the dimensions 313 x 30 x 131. Now let's look at the size of the
% other variable, clabel, and let's also print out its unique values
size(clabel)
unique(clabel)

% So clabel is a vector of size 313 x 1 and it contains only 1's and 2's. This
% number coincides with the first dimension of X, and it turns out that 313
% is the number of trials (called 'samples' in MVPA-Light) in the dataset.
% For each trial, clabel tells us which class the trial belongs to. 
% This dataset comes from an auditory oddball paradigm, and class 1 refers
% to trials wherein participants were presented a sound they were supposed
% to attend to (attended sounds), class 2 refers to trials wherein sounds
% were presented that the participant should not attend to (unattended
% sounds). Let's look at class labels for the first 20 trials
clabel(1:20)'

% we can see that the first 12 trials are class 2 (unattended sounds) where
% as trials 13-20 are of class 1 (attended sounds). To visualize the data,
% we can calculate the ERP for each class separately. Thus, we need to
% extract the indices of trials corresponding to each class
ix_attended = (clabel==1);    % logical array for selecting all class 1 trials 
ix_unattended = (clabel==2);  % logical array for selecting all class 2 trials 

% Let us print the number of trials in each class and the select the data
% from X:
fprintf('There is %d trials in class 1 (attended).\n', sum(ix_attended))
fprintf('There is %d trials in class 2 (unattended).\n', sum(ix_unattended))

X_attended = X(ix_attended, :, :);
X_unattended = X(ix_unattended, :, :);

% We have 102 trials in class 1 and 211 trials in class 2. This should
% coincide with the first dimension in X_attended and X_unattended, let's
% double check
size(X_attended)
size(X_unattended)

% To calculate the ERP, we now calculate the mean across the trials (first
% dimension). We use the squeeze function to the reduce the array from 3D
% to 2D, since we don't need the first dimension any more.
ERP_attended = squeeze(mean(X_attended, 1));
ERP_unattended = squeeze(mean(X_unattended, 1));
fprintf('Size of ERP_attended: [%d %d]\n', size(ERP_attended))
fprintf('Size of ERP_unattended: [%d %d]\n', size(ERP_unattended))

% Both ERPs now have the same size 30 (channels) x 131 (time points). Let
% us plot the ERP for channel Cz. To find the index of this channel, we
% need to use the channel labels (dat.label). We also need to define the
% time on the x-axis and will use dat.time for this.
ix = find(ismember(dat.label, 'Cz'));
figure
plot(dat.time, ERP_attended(ix, :))
hold all, grid on
plot(dat.time, ERP_unattended(ix, :))

legend({'Attended' 'Unattended'})
title('ERP at channel Cz')
xlabel('Time [s]'), ylabel('Amplitude [muV]')

%%%%%% EXERCISE 1 %%%%%%
% Now it's your turn: 
% Create another ERP plot, but this time select channel Fz. 
%%%%%%%%%%%%%%%%%%%%%%%%

% finally, let's plot the ERP for *all* channels. The plot will be more
% busy, but remember that each line now designates a different channel
figure
plot(dat.time, ERP_attended, 'r-')
hold all, grid on
plot(dat.time, ERP_unattended, 'b-')

title('ERP at all channels (red=attended, blue=unattended)')
xlabel('Time [s]'), ylabel('Amplitude [muV]')

% From this ERPs, it looks like the two classes are well-separated in the 
% 0.6 - 0.8 sec window. Our first analysis will focus just on this time
% window. To this end, we will average the time dimension in this window
% and discard the rest of the times. 
ival = find(dat.time >= 0.6 & dat.time <= 0.8);  % find the time points corresponding to 0.6-0.8 s

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival),3));
size(X)

% Note that now we went to a different representation of the data: X
% is now 313 (samples) x 30 (channels), and the channels will serve as our
% features. This is because classification is usually on the single-trial
% level, we only calculated the ERPs for visualization.

%% (2) Cross-validation and explanation of the cfg struct
rng(42) % fix the random seed to make the results replicable

% So far we have only loaded plotted the data. We did not do any MVPA
% analyses yet. In this section, we will get
% hands on with the toolbox. We use the output from the end of the previous
% section, the 2D [samples x channels] matrix X representing the EEG 
% activity in the 0.6-0.8s interval. Let's jump straight into it by passing
% X, clabel and the empty cfg struct to mv_classify:
cfg = [];
perf = mv_classify(cfg, X, clabel);

% There seems to be a lot going on here, so let's unpack the questions that
% might come up:
% 1. What happened? If we read the output on the console, we can figure out
% the following: mv_classify performed a cross-validation classification
% analysis using 5-fold cross-validation (k=5), 5 repetitions, using an 
% LDA classifier. This is simply the default behaviour if we don't specify
% anything else.

% 2. What is perf? Perf refers to 'performance metric', a measure of how
% good of a job the classifier did. By default, it calculates
% classification accuracy.
fprintf('Classification accuracy: %0.2f\n', perf)
% Hence, the classifier could distinguish both classes with an accuracy of
% 78% (0.78).

% 3. What does cfg do, it was empty after all?
% cfg controls all aspects of the classification analysis: choosing the
% classifier, a metric, preprocessing and defining the cross-validation. If
% it is unspecified, it is simply filled with default values. 
% For instance, let us change the classifier to Logistic Regression
% (logreg).

cfg = [];
cfg.classifier  = 'logreg';
perf = mv_classify(cfg, X, clabel);
fprintf('Classification accuracy using Logistic Regression: %0.2f\n', perf)

%%%%%% EXERCISE 2 %%%%%%
% Look at the available classifiers at 
% https://github.com/treder/MVPA-Light/blob/master/README.md#classifiers
% Run the classification again, this time using a Naive Bayes classifier.
%%%%%%%%%%%%%%%%%%%%%%%%

% Now we know how to set a classifier, let's see how we can change the
% metric that we want to be calculated.  Let's go for area under the ROC
% curve (auc) instead of accuracy. 
cfg = [];
cfg.metric      = 'auc';
perf = mv_classify(cfg, X, clabel);
fprintf('AUC: %0.2f\n', perf)

% We can also calculate both AUC and accuracy at the same time using a cell
% array. Now perf will be a cell array, the first value is the AUC value,
% the second value is classification accuracy. Since we do not specify a
% classifier, the default classifier (LDA) is used again.
cfg = [];
cfg.metric      = {'auc', 'accuracy'};
perf = mv_classify(cfg, X, clabel);

perf

%%%%%% EXERCISE 3 %%%%%%
% Look at the available classification metrics at 
% https://github.com/treder/MVPA-Light/blob/master/README.md#metrics
% Do the classification again, this time calculating precision and recall.
%%%%%%%%%%%%%%%%%%%%%%%%

% We know now how to define the classifier and the performance metric. We
% still need to understand how to change the cross-validation scheme. Let us
% perform k-fold cross-validation with 10 folds (i.e. 10-fold
% cross-validation) and 2 repetitions. Note how the output on the console 
% changes.
cfg = [];
cfg.k           = 10;
cfg.repeat      = 2;
perf = mv_classify(cfg, X, clabel);

%%%%%% EXERCISE 4 %%%%%%
% Look at the description of cross-validation at 
% https://github.com/treder/MVPA-Light/blob/master/README.md#cv
% Do the classification again, but instead of k-fold cross-validation use
% leave-one-out (leaveout) cross-validation.
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% EXERCISE 5 %%%%%%
% This is a conceptual question: why is it useful to have multiple 
% repetitions of the cross-validation analysis? Why don't we just run it
% once?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (3) Classification of data with a time dimension
% If the data X is three-dimensional e.g. [samples x channels x time points], 
% we can perform a classification for every time point separately. This is 
% useful e.g. for event-related experimental designs.
% Let's go back to the original data then, which had 313 samples x 30 EEG
% channels x 131 time points.
X = dat.trial;
size(X)

% We can again use mv_classify. X now has 3 dimensions, but mv_classify
% simply loops over any additional dimension. 
cfg = [];
perf = mv_classify(cfg, X, clabel);

% Looking at the size of perf, we now obtained a classification accuracy 
% for each of the 131 time points:
size(perf)

% When we plot it, we can see that classification performance is high 
% between 0.2 - 0.8 sec.
close all
plot(dat.time, perf, 'o-')
grid on
xlabel('Time'), ylabel('Accuracy')
title('Classification across time')


% For [samples x features x time] data, MVPA-Light also has a specialized
% function called mv_classify_across_time. It does the same thing as
% mv_classify in this case, but it can be faster so you are recommended to
% use it in these cases. The only visible difference is that in the output
% the dimensions are now labeled as 'samples', 'features', and 'time
% points'. Both mv_classify and mv_classify_across_time use the same type
% of parameters for the cfg struct
cfg = [];
perf = mv_classify_across_time(cfg, X, clabel);

%%%%%% EXERCISE 6 %%%%%%
% Let's put together everything we learned so far: Use
% mv_classify_across_time with a Logistic Regression classifier and
% 20-fold cross-validation with 1 repetition. Use Cohen's kappa as a 
% classification metric. Plot the result.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (4) Time generalization (time x time classification): 
% Sometimes we want to train the classifier at a given time point t1 and 
% test it at *all* time points t2 in the trial. If we repeat this for every
% combination of training and test time points, we obtain a [time x time] 
% matrix of results. This is also known as the temporal generalization
% method (see paper by King & Dehaene, 2014;
% https://pubmed.ncbi.nlm.nih.gov/24593982/).
% Generalization can be performed with mv_classify, but if the data
% dimensions are [samples x features x time points] and you want to
% calculate time generalization, you can use the more specialized function
% mv_classify_timextime. Again, it acts exactly like mv_classify, it simply
% has some useful presets.

cfg = [];
cfg.metric      = 'auc';
auc = mv_classify_timextime(cfg, dat.trial, clabel);

% plot the resultant image
close all
imagesc(dat.time, dat.time, auc)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Training time'), xlabel('Test time')
% This looks more complicated. Now the classification accuracy is coded as
% a color, and the colorbar indicates which classification accuracy
% corresponds to which color. The y-axis specifies the time at which the
% classifier was trained. The x-axis specifies the time at which it was
% tested.
% A striking feature in the image is a 'block' in the range 
% 0.4 - 0.8s suggesting that there is a stable representation within this
% time. For instance, a classifier trained at 0.4s can still decode
% relatively well at 0.6 or 0.8s, so the representations must be very
% similar within this time period. 

% Note that mv_classify_timextime is mostly a convenience function. Time
% generalization can be performed using mv_classify, too. To see this, let
% us repeat this analysis using mv_classify. mv_classify can also 
% generalize over any dimension. Since our data is [samples x features x
% time] and we want to generalize over the 3rd dimension (time), we need to
% set cfg.generalization_dimension = 3.
cfg = [];
cfg.metric                      = 'auc';
cfg.generalization_dimension    = 3;
auc2 = mv_classify(cfg, dat.trial, clabel);

%%%%%% EXERCISE 7 %%%%%%
% Repeat the time x time classification without cross-validation
% (cv='none'). What do you notice?
%%%%%%%%%%%%%%%%%%%%%%%%

% Generalization with two datasets (aka cross decoding): 
% So far we trained and tested on the
% same dataset. However, nothing stops us from training on one dataset and
% testing on the other dataset. This can be useful e.g. in experiments with
% different experimental conditions (eg memory encoding and memory
% retrieval) where one may want to investigate whether representations in
% the first phase re-occur in the second phase. 
%
% We do not have such example data, so instead we will do cross-participant
% classification: train on the data of participant 1, test on the data of
% participant 2
[dat1, clabel1] = load_example_data('epoched1');  % participant 1
[dat2, clabel2] = load_example_data('epoched2');  % participant 2

% To perform this, we can pass the second dataset and the second class
% label vector as extra parameters to the function call. Note that no
% cross-validation is performed since the datasets are independent. It is
% useful to use AUC instead of accuracy here, because AUC is not affected
% by differences in offset and scaling that the two datasets might have.
cfg =  [];
cfg.classifier = 'lda';
cfg.metric     = 'auc';

cross_auc = mv_classify_timextime(cfg, dat1.trial, clabel1, dat2.trial, clabel2);

close all
imagesc(dat2.time, dat1.time, cross_auc)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Participant 1 time'), xlabel('Participant 2 time')
% The result does not look very convincing. On the positive side, training
% at about 0.4s and testing in the same range (0.3-0.5s) yields a
% performance of close to 0.8 AUC which suggests that what the classifier
% learnt in participant 1 transfers to participant 2.
% However, there is a strange result as well: training at
% about -0.18s also allows successful decoding in the 0.3-0.5s range. It 
% is likely that this effect is coincidental and that it disappears when
% the analysis is repeated for different combinations of participants
% and the average across these analyses is plotted.

%%%%%% EXERCISE 8 %%%%%%
% Repeat the cross-classification but reverse the order of the
% participants: train on participant 2 and test on participant 1. Do you 
% expect the same result?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (5) Plotting results
% So far, we have plotted the results by hand using Matlab's plot
% function. For a quick and dirty visualization, MVPA-Light has a function
% called mv_plot_result. It plots the results and nicely lays out the axes
% for us. To be able to use it, we need the result struct, which is simply
% the second output argument of any classification function.
% Let's test the visualization for 2D data first by selecting the 100-th
% time point
X = mean(dat.trial(:,:,100), 3);

cfg = [];
cfg.metric = 'auc';
[~, result] = mv_classify(cfg, X, clabel);

% Now call it passing result as an input argument. We will obtain a barplot
% representing the AUC. The height of the bar is equal to the value of
% perf. The errorbar is the standard deviation across folds and
% repetitions, an heuristic marker of how variable the performance measure
% is for different test sets.
close all
mv_plot_result(result)

% Next, let us perform classification across time. The result will be a
% time series of AUCs. The line represents the mean (equal to the values 
% in perf), the shaded area is again the standard deviation across
% folds/repeats.
cfg = [];
cfg.metric = 'auc';
[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);

mv_plot_result(result)

% The x-axis depicts the sample number, not the actual time points. To get the
% x-axis in seconds, we can provide the time points as an extra argument to
% the function call.
mv_plot_result(result, dat.time)

% Lastly, let us try time generalization (time x time classification). For
% the resultant plot, both the x-axis and the y-axis need to be specified.
% Therefore, we pass the parameter dat.time twice to get both axes in
% seconds.
cfg = [];
cfg.metric = 'auc';
[~, result] = mv_classify_timextime(cfg, dat.trial, clabel);

g = mv_plot_result(result, dat.time, dat.time);

% the output argument g contains some handles to the graphical elements
% which may be convenient when customizing the layout of the plots. For
% instance, let us change the font size of the title:
set(g.ax.title, 'FontSize', 28)

%%%%%% EXERCISE 9 %%%%%%
% What happens when you call mv_plot_result and you calculated multiple
% metrics at once e.g. precision, recall and F1 score?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (6) Searchlight analysis
% 
% Searchlight analysis aims to combine the advantages of multivariate 
% approaches such as classifiers (high statistical power) with the
% advantages of univariate approaches (high localizability and therefore
% interpretability). It strikes a balance between localization and
% statistical power. Imagine you not only want to know whether two classes
% (eg face vs house images) can be discriminated but also which EEG
% electrodes are most important for this classification. Searchlight
% analysis is one approach to localize multivariate effects.
%
% A good reference for searchlight analysis is the following paper:
% Kriegeskorte N, Goebel R, Bandettini P. Information-based functional brain mapping. 
% Proc Natl Acad Sci U S A. 2006 Mar 7;103(10):3863-8. doi: 10.1073/pnas.0600244103. 
% Epub 2006 Feb 28. PMID: 16537458; PMCID: PMC1383651.
% https://pubmed.ncbi.nlm.nih.gov/16537458/

% Load the data again, but this time we will keep a third output argument
% chans which describes the layout of the EEG channels. It will be useful
% for plotting the result.
[dat, clabel, chans] = load_example_data('epoched3');

% We start off with our 3D data which is [samples x electrodes x time
% points]. We want to know which electrodes contribute most to the
% classification outcome. To this end, we will now perform a classification
% for each electrode separately. As features, we will use all time points.
% This can easily be realized in mv_classify by defining which dimension 
% represents the features: by default this is dimension 2 (electrodes), so
% we will set it to dimension 3 (time points).
cfg = [];
cfg.metric              = 'auc';
cfg.feature_dimension   = 3;        % now the time points act as features
cfg.dimension_names     = {'samples' 'electrodes' 'time points'}; % name the dimensions for nicer output
[perf, result] = mv_classify(cfg, dat.trial, clabel);

% The resultant plot shows that each electrode contains information about
% the classes
mv_plot_result(result)

% However, the visualization is not great: It would be nice to see the
% result as a spatial topography. We can use the function
% mv_plot_topography for this and we will need the chans.pos field which
% represents the 2D coordinates of the electrodes.
cfg_plot = [];
cfg_plot.outline = chans.outline;
figure
mv_plot_topography(cfg_plot, perf, chans.pos);
colormap jet
% This plot contains the same information as the line plot produced by
% mv_plot_result, but its spatial layout makes it easier to interpret the
% result: we can now appreciate that occipital and left parietal and 
% central electrodes contribute most to classification performance.

%%%%%% EXERCISE 10 %%%%%%
% Classification across time (section 3) can also be interpreted as a type
% of searchlight analysis: instead of classifying all electrodes separately
% (using time points as features), in classification across time we
% classify each time point separately (using electrodes as features). 
% Change the value of cfg.feature_dimension in the searchlight analysis to
% perform classification across time.
%%%%%%%%%%%%%%%%%%%%%%%%

% So far, we used classified electrodes separately. We can increase the size
% of our searchlight by considering an electrode and its direct neighbours
% for classification, and then attributing the classification result to
% this electrode. We first must find out which electrodes are neighbours of
% each other. To this end, build a distance matrix representing pair-wise 
% distances between electrodes according to the chans.pos field.
nb_mat = squareform(pdist(chans.pos));
% Eg nb_mat(2,5) is the 2D distance between electrodes 2 and 5

% From this matrix, we can define neighbours by defining a cutoff value. All
% electrodes closer to each other than the cutoff value are considered as
% neighbours.
cutoff = 0.2;
electrode_neighbours =  (nb_mat < cutoff);

% Looking at the upper left corner of the matrix, we can eg see that electrode 1
% (represented by row 1) is of course its own neighbour (column 1) but
% electrode 3 (col 3) is also its neighbour. In fact, these are the only
% two neighbours for this electrode.
electrode_neighbours(1:5,1:10)

% If we sum each row we get the number of neighbours each electrode has
sum(electrode_neighbours,2)'
% Eg the 1st electrode has 2 neighbours, and the 3rd electrode has 5
% neighbours.

% Now let us rerun the analysis, specifying the cfg.neighbours field
cfg = [];
cfg.metric              = 'auc';
cfg.feature_dimension   = 3;
cfg.neighbours          = electrode_neighbours;  % pass the neighbours matrix
cfg.dimension_names     = {'samples' 'electrodes' 'time points'};
perf = mv_classify(cfg, dat.trial, clabel);

% plot the result
cfg_plot = [];
cfg_plot.outline = chans.outline;
figure
mv_plot_topography(cfg_plot, perf, chans.pos);
colormap jet
title('With neighbours')
% We see that now the best classification performance is achieved for
% left central electrodes. Also the result looks smoother than without
% using neighbours. This makes sense because we are now using each
% electrode and its surrounding electrodes for classification.

% Now let us get back to classification across time and abstract the notion
% of neighbours: what is the neighbour of a given time point?
% The neighbour of a time point can be defined as the
% immediately preceding and following time points. For instance, the
% neighbours of the time point 100 are the time points 99 and 101. 
% We can encode this is the neighbours matrix by specifying a diagonal
% matrix that contains the main diagonal (each time point is its own
% neighbour) and the two neighbouring off-diagonals (each time point
% neighbours the immediately preceding and following time points).
n_time = numel(dat.time);
I = eye(n_time); % diagonal: it specifies that each time point is a neighbour of itself
offdiag = diag(ones(n_time-1, 1),1) + diag(ones(n_time-1, 1),-1); % preceding and following time points
time_neighbours      = I + offdiag;

% To understand the structure of the matrix, let's look at its upper left
% corner
time_neighbours(1:5,1:5)
% Time point 1 (row 1) has two neighbours: itself (col 1) and the following
% time point (col 2). There is no preceding time point.
% Time point 2 (row 2) has three neighbours: the preceding time point (col
% 1), itself (col 2), and the following time point (col 3). 

time_neighbours(end-5:end, end-5:end)
% Looking at the bottom right corner of the matrix, we see that the same
% pattern continues for all other time points except for the last time
% point, which has no following time point (only a preceding time point).
% Let's now pass this matrix to mv_classify and repeat the classification:

cfg.feature_dimension   = 2;
cfg.neighbours          = time_neighbours;
[perf, result] = mv_classify(cfg, dat.trial, clabel);

% The result looks slightly smoother (less wiggly line)
mv_plot_result(result, dat.time)

%%%%%% EXERCISE 11 %%%%%%
% We performed searchlight analyses across both electrodes and time points.
% Now it's time to put everything together: Can you use my_classify to define a
% searchlight analysis across both electrodes and time points? 
% What are the dimensions of the result?
% Hint: you have to leave the feature dimension empty, and you need to
% provide both neighbours matrices as a cell array.
%%%%%%%%%%%%%%%%%%%%%%%%


%% (7) Hyperparameters
% Hyperparameters control the behaviour of the classifiers. They can be
% used to e.g. select the kernel for SVM and set the regularization
% strength in LDA. MVPA-Light is designed such that the standard settings
% work for many classification problems, but for more fine grained control
% of the classifier you may want to control them yourself. 
% Hyperparameters are classifier specific, so you may want to inspect a
% classifier's train function for a description of the hyperparameters.
% Let's start with SVM.
help train_svm

% We can see that the kernel hyperparameter can be used to select the
% kernel function. When using a high-level function such as
% mv_classify_across_time, the cfg.hyperparameter field is used to specify
% hyperparameters. Let us set the kernel to rbf

cfg = [];
cfg.classifier              = 'svm';
cfg.hyperparameter          = [];
cfg.hyperparameter.kernel   = 'rbf'; 
perf_rbf = mv_classify(cfg, X, clabel);

%%%%%% EXERCISE 12 %%%%%%
% Repeat the analysis, but select a polynomial kernel of degree 3.
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% EXERCISE 13 %%%%%%
% In LDA, the shrinkage regularization hyperparameter is by default
% estimated automatically. Can you find the hyperparameter than controls
% the regularization strength and set it to 0.1?
%%%%%%%%%%%%%%%%%%%%%%%%

% Congrats, you made it to the end! You can embark on your own MVPA 
% adventures now or check out one of the other tutorials.

%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
% We only need to find the index of channel Fz, the rest of the code is 
% the same as before. The result looks slightly different but we can still
% see a difference between the two classes.
ix = find(ismember(dat.label, 'Fz'));

figure
plot(dat.time, ERP_attended(ix, :))
hold all, grid on
plot(dat.time, ERP_unattended(ix, :))

legend({'Attended' 'Unattended'})
title('ERP at channel Fz')
xlabel('Time [s]'), ylabel('Amplitude [muV]')

%% SOLUTION TO EXERCISE 2
% Looking in the Readme file, we can see that the Naive Bayes classifier
% is denoted as naive_bayes 
cfg = [];
cfg.classifier  = 'naive_bayes';

perf = mv_classify(cfg, X, clabel);

% We can see that in this case the accuracy is lower for Naive Bayes than 
% for LDA or Logistic Regression
perf

%% SOLUTION TO EXERCISE 3
% Looking at the Readme file, we see that precision and recall are
% simply denoted as 'precision' and 'recall'
cfg = [];
cfg.metric = {'precision', 'recall'};
perf = mv_classify(cfg, X, clabel);

fprintf('Precision = %0.2f, Recall = %0.2f\n', perf{:})

%% SOLUTION TO EXERCISE 4
% Leave-one-out cross-validation is called 'leaveout' and we need to set
% the cv field to define it. Now, you will see that we have 313 folds and
% only one repetition: In leave-one-out cross-validation, each of the 313
% samples is held out once (giving us 313 folds). It does not make sense to
% repeat the cross-validation more than once, since there is no randomness
% in assigning samples to test folds any more (every sample is in a test
% fold once).
cfg = [];
cfg.cv      = 'leaveout';
perf = mv_classify(cfg, X, clabel);

%% SOLUTION TO EXERCISE 5
% Cross-validation relies on random assignments of samples into folds. This
% randomness leads to some variability in the outcome. For instance, let's
% assume you find a AUC of 0.78. When you rerun the analysis it changes to
% 0.75 or 0.82. Having multiple repetitions and averaging across them
% reduces the variability of your metric, but it comes at higher
% computational costs.
%
% We can check this empirically by first running a classification analysis
% several times with 1 repeat and then comparing it to 5 repeats. For the 5
% repeats, the variability (=standard deviation) of the classification
% scores should be smaller than for the 1 repeat case, illustrating that
% it is more stable/replicable.

one_repeat = zeros(10,1);
five_repeats = zeros(10,1);

% for simplicity and speed, we reduce the data to 2D by selecting the
% sample at the middle time point
X = dat.trial(:,:,floor(end/2));

% one repeat, run the analysis 10 times
cfg = [];
cfg.repeat      = 1;
cfg.feedback    = 0;  % suppress output
for ii=1:10
    one_repeat(ii) = mv_classify(cfg, X, clabel);
end

% five repeats, run the analysis 10 times
cfg = [];
cfg.repeat      = 5;
cfg.feedback    = 0;  % suppress output
for ii=1:10
    five_repeats(ii) = mv_classify(cfg, X, clabel);
end

one_repeat'
five_repeats'

fprintf('Standard deviation of metric for one repeat: %0.5f\n', std(one_repeat))
fprintf('Standard deviation of metric for five repeats: %0.5f\n', std(five_repeats))
% We can see that the standard deviation of the metric becomes much smaller
% if we have more repeats. This suggests that our metric is more stable
% now.

%% SOLUTION TO EXERCISE 6
% We simply set cfg.classifier, cfg.metric, cfg.k and cfg.repeat to the
% required values and then perform the classification.
X = dat.trial;

cfg = [];
cfg.classifier      = 'logreg';
cfg.metric          = 'kappa';
cfg.k               = 20;
cfg.repeat          = 1;
perf = mv_classify_across_time(cfg, X, clabel);

close all
plot(dat.time, perf, 'o-')
grid on
xlabel('Time'), ylabel(cfg.metric)
title('Classification across time')

%% SOLUTION TO EXERCISE 7
cfg = [];
cfg.cv          = 'none';
cfg.metric      = 'auc';
auc = mv_classify_timextime(cfg, dat.trial, clabel);

% When cross-validation is turned off, most of the result looks very
% similar. However, a diagonal appears running from the bottomn left to the 
% top right. This is because, without cross-validation, we get some
% overfitting (this is why we have good classification performance even in 
% the pre-stimulus phase). This is especially evident on the diagonal
% because here train and test samples are identical.
% Cross-validation prevents this from happening.
figure
imagesc(dat.time, dat.time, auc)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Training time'), xlabel('Test time')
title('No cross-validation')

%% SOLUTION TO EXERCISE 8
% We just need to reverse the order of the arguments, feeding in dat2 and
% clabel2 first and then dat1 and clabel1. The result is not the same, but
% this is not to be expected: in cross-decoding we identify discriminative
% patterns in the first dataset and then look for them in the second
% dataset. The discriminative patterns for two subjects are likely
% not identical.
cfg = [];
cfg.classifier = 'lda';
cfg.metric     = 'auc';

cross_auc2 = mv_classify_timextime(cfg, dat2.trial, clabel2, dat1.trial, clabel1);

% We find a different pattern. Most obviously training at around 0.7 s in
% participant 2 seems to yield above chance performance in participant 1 
% in the 0.5 - 1 s window.
figure
imagesc(dat2.time, dat1.time, cross_auc2)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Participant 2 time'), xlabel('Participant 1 time')

%% SOLUTION TO EXERCISE 9
% With multiple metrics, the mv_plot_result function simply creates
% multiple figures, one for each metric. We will illustrate this here for
% the different types of plots we generated in this tutorial
cfg = [];
cfg.metric = {'precision' 'recall' 'f1'};

close all

% Classify a single time point: we obtain three bar plots. The y-axis
% label shows which of the metrics is depicted.
X = mean(dat.trial(:,:,100), 3);
[~, result] = mv_classify(cfg, X, clabel);
mv_plot_result(result)

% Classification across time: we obtain three line plots. Again the y-axis
% shows which of the metrics is depicted.
[~, result] = mv_classify_across_time(cfg, dat.trial, clabel);
mv_plot_result(result, dat.time)

% Classify time x time: we obtain three images. The name of the metric 
% appears on top of the colorbar.
[~, result] = mv_classify_timextime(cfg, dat.trial, clabel);
mv_plot_result(result, dat.time, dat.time);

%% SOLUTION TO EXERCISE 10
% It is very simple, we just have to set cfg.feature_dimension = 2 instead
% of 3. This will tell mv_classify that we use electrodes as features, and
% it will automatically loop over the 3rd dimension (time points). This is
% actually the default but we will set it here manually, just to be extra
% clear.
cfg = [];
cfg.metric = 'auc';
cfg.feature_dimension = 2;
cfg.dimension_names = {'samples' 'electrodes' 'time points'}; % name the dimensions for nicer output
[~, result] = mv_classify(cfg, dat.trial, clabel);

% now the result is for each of the 131 time points
mv_plot_result(result, dat.time)

%% SOLUTION TO EXERCISE 11
% We have to leave the feature dimension empty since the searchlight is to
% be performed across both electrodes and time points. We furthermore need
% to pass the electrode_neighbours and time_neighbours matrices as a cell
% array. The order they are passed in must be equal to the order the
% dimensions appear in the data.
cfg = [];
cfg.feature_dimension   = [];
cfg.neighbours          = {electrode_neighbours, time_neighbours};  % pass both neighbours matrices
cfg.dimension_names     = {'samples' 'electrodes' 'time points'};
[perf, result] = mv_classify(cfg, dat.trial, clabel);

% The result is of size 30 x 131, 30 electrodes and 131 time points
size(perf)

% Since the result is 2D, it will be displayed as an image
h = mv_plot_result(result, dat.time);

% We now have an image of electrodes x time: this shows us both which
% electrodes contribute to classification and when they do. Let us add the
% electrode labels on the y-axis to make the plot more informative:
set(h.ax.ax,'YTick', 1:numel(dat.label), 'YTickLabel', dat.label)

%% SOLUTION TO EXERCISE 12
cfg = [];
cfg.classifier              = 'svm';
cfg.hyperparameter          = [];
cfg.hyperparameter.kernel   = 'polynomial'; 
cfg.hyperparameter.degree   = 3;
perf_poly = mv_classify(cfg, X, clabel);

%% SOLUTION TO EXERCISE 13
% Let's look at the help first to see a list of hyperparameters
help train_lda

% We see that the parameter lambda controls the regularization strength
cfg = [];
cfg.classifier              = 'lda';
cfg.hyperparameter          = [];
cfg.hyperparameter.lambda   = 0.1;
perf = mv_classify(cfg, X, clabel);
