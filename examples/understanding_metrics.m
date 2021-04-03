% UNDERSTANDING METRICS
%
% The purpose of this tutorial is to gain a better understanding of
% the different classification and regression metrics in MVPA-Light as well
% as the different types of outputs that classifiers produce.
%
% Contents:
% (1) Classification: Relationship between dvals (decision values), accuracy, and raw
%     classifier output
% (2) Classification: Looking at three types of raw classifier output: 
%     class labels, dvals, and probabilities
% (3) Regression: Relationship between the raw model output (predictions)
%     and the metrics MAE and MSE
% (4) Classification: Compare classification metrics
%
% Note: If you are new to working with MVPA-Light, make sure that you
% complete the introductory tutorials first:
% - getting_started_with_classification
% - getting_started_with_regression
% They are found in the same folder as this tutorial.

close all
clear

%% Loading example data
[dat, clabel] = load_example_data('epoched3');
X = dat.trial;

%% (1) Classification: Relationship between dvals (decision values), accuracy, and raw classifier output

% In this section we investigate the relationship between dvals,
% accuracy and the raw classifier output. 
% To keep the results simple, we will use just 1 hold out set as
% cross-validation approach. We will then extract classification accuracy
% ('accuracy'), decision values ('dval'), and raw classifier output ('none').
% Note that the raw classifier output can be single-trial predicted class
% labels, or decision values, or probabilities. We will use a LDA classifier 
% which, by default, produces decision values. To explicitly control the
% raw classifier output, cfg.output_type can be set.
cfg = [];
cfg.metric      = {'accuracy' 'dval' 'none'};
cfg.cv          = 'holdout';
cfg.p           = 0.5;
cfg.repeat      = 2;
cfg.output_type = 'dval';  % make sure that the raw classifier output is dvals not class labels
cfg.classifier  = 'lda';

[perf, result] = mv_classify_across_time(cfg, X, clabel);

% Before looking at the results let us recall the dimensions of the data
% which are [313, 30, 131], that is 313 samples, 30 channels, 131 time points 
size(X)

% Let's look at perf now:
% Recall perf is a cell array with three elements because we requested 3
% metrics. So perf{1} corresponds to 'accuracy',
% perf{2} corrresponds to 'dval', and perf{3} to 'none' (raw classifier outputs). 
% If we print the cell array we can have a closer look at the dimensions
perf

% Let us focus at perf{1} and perf{2} for now. We see that size(perf{1}) is
% is [131,1]
size(perf{1})

% whereas size(perf{2}) is [131,2]. 
size(perf{2}) 

% So for 'accuracy' we a vector of accuracy values, one accuracy for each
% time point. However, for 'dval' we get two such vectors. Let us visualize
% the result to see why
close all
mv_plot_result(result)

% Figure 1 shows accuracy: a single line, since it takes both classes into
% account simultaneously.
% Figure 2 shows dval: we get two lines since the average dval is
% calculated for each class separately.
% Figure 3 shows the raw classifier output: Each dot represents a single
% sample, and the dots are colored according to which class the sample
% belongs to. If you average the dvals in each class at each x-value, you
% obtain the the dval metric shown in the Figure 2. In other words, Figure
% 2 shows the class-wise average of the values in Figure 3.

%%%%%% EXERCISE 1 %%%%%%
% Looking at the dimensions of perf for the 'none' metric, we get 
% size(perf{3}) = [1, 1, 131]? Why is it not [131, 1] like the accuracy
% metric? What do the first two dimensions encode?
% Hint: rerun the analysis with 3-fold cross-validation and 2 repetitions
% and look at the size again.
%%%%%%%%%%%%%%%%%%%%%%%%


%% (2) Classification: Looking at three types of raw classifier output: class labels, dvals, and probabilities

% Classifiers such as LDA can produce different types of outputs: class
% labels, dvals, and probabilities. These outputs are not the same as
% metrics: classifier outputs are the raw outputs produced by a classifier,
% whereas metrics are summaries calculated on basis of these raw outputs. In the
% previous exercise, we have seen that cfg.output_type can be used to
% select the type of raw output. 

cfg = [];
cfg.metric      = 'none';
cfg.output_type = 'clabel';  % 'clabel' is also the default for the 'none' metric
cfg.cv          = 'none';    % cv = 'none' means the entire training set serves as test set.
cfg.classifier  = 'lda';
    
[perf_clabel, result_clabel] = mv_classify_across_time(cfg, X, clabel);

% Now let's set the output type to dval and compute the output
cfg.output_type = 'dval';
[perf_dval, result_dval] = mv_classify_across_time(cfg, X, clabel);

% Now let's set the output type to prob
cfg.output_type = 'prob';
% To get probabilities, we must also set LDA's .prob hyperparameter to one,
% because in order to calculate probabilities a multivariate Gaussian
% distribution needs to be estimated in the training phase
cfg.hyperparameter = [];
cfg.hyperparameter.prob = 1;
[perf_prob, result_prob] = mv_classify_across_time(cfg, X, clabel);

% Let's print the first 5 elements from each of the results using the
% 101-st time point (corresponding to  dat.time(101) = 0.7s post-stimulus))
% Since we set cv = 'none' there is no random folds, hence the first 5 elements 
% in each result correspond to the first 5 samples in the data.
% (perf has two leading singleton dimensions because technically 
% the data belongs to the 1st repetition and 1st fold)
perf_clabel{1,1,100}(1:5)
perf_dval{1,1,100}(1:5)
perf_prob{1,1,100}(1:5)

% let us also compare the first 5 true class labels
clabel(1:5)

% We see that
% - the first 5 samples are from class 2 (see clabel)
% - the classifier predicts 4 of these samples correctly (see perf_clabel)
% - we get positive dvals for samples predicted as class 1 and negative dvals for samples predicted as class 2 (see perf_dval)
% - we get probabilities >0.5 for samples predicted as class 1 and probabilities <=0.5 
%   for samples predicted as class 2 (see perf_prob). Hence the
%   interpretation of the probability is "probability that the sample
%   belongs to class 1". The threshold of 0.5 is defined as the cutoff.
% This also shows that we can directly calculate the class labels from
% either dvals or probabilities.

%%%%%% EXERCISE 2 %%%%%%
% Use the mv_plot_result function to plot the three result structs. Can you
% interpret the plots? One of the three plots is not very useful, which
% one?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (3) Regression: Relationship between the raw model output (predictions) and the metrics MAE and MSE

% In this section we will look into how the output of a regression model
% relates to the metrics MAE and MSE. We will use the same EEG data we used
% for classification, but to simplify the analysis we will focus on a
% single time point. Let us perform regression on all time points first and
% then select the time point that gives us the best result. 
% As target values, we will use the trial number:
y = (1:size(dat.trial,1))';

% Train a Kernel Ridge regression model and calculate MAE
cfg = [];
cfg.model                   = 'ridge';
cfg.hyperparameter          = [];
cfg.hyperparameter.lambda   = 0.1;
cfg.metric                  = 'mae';

% Since we want to predict the trial number both train and test sets should
% ideally contain early, middle, and late trials. An easy way to achieve
% this is to predefine two folds: the first fold contains trials with
% uneven trials numbers (1, 3, 5, ...) and the second folds contains the
% even trials.
cfg.cv                      = 'predefined';    
fold = ones(numel(y), 1); 
fold(2:2:end) = 2; % even trials are designated as fold 2
cfg.fold                    = fold;


[perf, result] = mv_regress(cfg, dat.trial, y);

% plot result and mark point with lowest MAE
mv_plot_result(result, dat.time)
hold on
[~, min_ix] = min(perf);  % index of time point with lowest MAE
plot(dat.time(min_ix), perf(min_ix), 'r.', 'MarkerSize', 36) % mark point with the lowest MAE in red

% From the plot we see that at time index min_ix the error is the lowest. Let 
% us select this time point before we proceed
X = squeeze(dat.trial(:, :, min_ix));

% For this time point, train another model and calculate MAE, MSE and the raw predictions
cfg.metric         = {'mae' 'mse' 'none'};
[perf, result] = mv_regress(cfg, X, y);
perf

% perf{1} and perf{2} represent MAE and MSE. perf{3} contains the raw
% predictions. It is a nested cell array with the results for the two folds
% The first fold perf{3}{1} represents the uneven trial numbers and
% perf{3}{2} represents even trial numbers.  Let us unpack these two into a
% variable pred
pred = zeros(numel(y), 1);
pred(1:2:end) = perf{3}{1};
pred(2:2:end) = perf{3}{2};
% Now pred contains the predictions in order, i.e. pred(i) is the
% prediction for the i-th trial

close all
plot(1:numel(y), pred)
grid on
xlim([0, 313]), ylim([0, 313])
xlabel('Trial number')
ylabel('Prediction trial number')
% The x-axis corresponds to the trial number, the y-axis to the predicted
% trial number. Although prediction performance appears low, there is a slight 
% positive trend as x increases suggesting that the model is partly able to
% predict the trial number from the EEG. 

%%%%%% EXERCISE 3 %%%%%%
% How can the predictions in pred be used to calculate perf{1} (representing MAE) 
% and perf{2} (representing MSE)?
% Hint: Consider the two folds separately, then use a weighted average.
%%%%%%%%%%%%%%%%%%%%%%%%


%% (4) Compare classification metrics

% In this section we calculate a whole range of classification metrics and
% display them in a single plot. The plot will indicate that the metrics
% tend to be correlated with each other.

% Perform classification across time for the different metrics. 
cfg = [];
cfg.metric      = {'accuracy' 'f1' 'kappa' 'precision' 'recall' 'auc' 'tval'};
perf = mv_classify_across_time(cfg, dat.trial, clabel);

% transform cell array to 2D array
perf = cat(2, perf{:});
perf(:, end) = perf(:, end)/6; % tval has a much different scaling from the other metrics, to fit it into the same plot we scale it down

% plot the means of all metrics
close all
plot(dat.time, perf)
grid on
legend(cfg.metric)

% While all the metrics have different interpretations and they can
% diverge, we often find that they are highly correlated. In fact, looking
% at the correlation matrix we see that all metrics have a correlation >0.9
fprintf('Correlation matrix =\n')
disp(corr(perf))
% The metrics are expected to diverge more when e.g. the data is strongly
% unbalanced or the classifier is clearly biased towards one of the
% classes.

% Congrats, you finished the tutorial!

%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
cfg = [];
cfg.metric      = {'accuracy' 'dval' 'none'};
cfg.cv          = 'kfold';
cfg.k           = 3;
cfg.repeat      = 2;
    
[perf, result] = mv_classify_across_time(cfg, dat.trial, clabel);

perf
% Now we see that size(perf{3}) = [2, 3, 131]. So the first dimension
% represents the number of repetitions (2), the second the number of test
% folds (3), and the third is the number of time points. For metric='none'
% we get separate results for each repetition and fold because the results
% are not averaged across folds/repetitions.

%% SOLUTION TO EXERCISE 2
% Let's start with plotting the result based on the clabel outputs.
mv_plot_result(result_clabel)
% We see only two horizontal lines of dots at y=1 and y=2: this is 
% because we are plotting the class labels which have the values 1 and 2.
% Furthermore, class 1 dots are plotted first, followed by class 2 dots.
% Class 2 dots are superimposed on class 1 which is why we can only see one
% class. This visualization is not very useful, it would look very much the
% same even if the classifier was classifying at random.

% Let us look at the dvals now. We can add dat.time as a second argument,
% this makes sure that the x-axis correspond to the time within the epoch:
mv_plot_result(result_dval, dat.time)
% We can see both class 1 and 2. As expected, class 1 tends to be positive
% and class 2 tends to be negative. The y-axis now represents dvals. The
% scaling of the y-axis depends on the type of classifier and potentially
% the scaling of the data, but the relative relationships between the
% classes are more important here.

% Let us now compare these dvals to the probability values:
mv_plot_result(result_prob, dat.time)
% Now the y-axis is confined to [0, 1]. It represents the probability of a 
% sample belonging to class 1. Naturally, class 1 samples tend to have a 
% higher probability than class 2 samples. Moreover, comparing probabilities 
% to dvals we can see a good correspondence: for each time point wherein 
% the dvals for class 1 vs 2 dvals are further apart, their corresponding 
% probabilities are further apart, too. It is especially at these time
% points that discriminability is high.

%% SOLUTION TO EXERCISE 3
% Let's first reproduce the result in example 3
% Train a Kernel Ridge regression model and calculate MAE
cfg = [];
cfg.model                   = 'ridge';
cfg.hyperparameter          = [];
cfg.hyperparameter.lambda   = 0.1;
cfg.metric                  = 'mae';
cfg.cv                      = 'predefined';    
fold = ones(numel(y), 1); 
fold(2:2:end) = 2; 
cfg.fold                    = fold;
cfg.metric                  = {'mae' 'mse' 'none'};
[perf, result] = mv_regress(cfg, X, y);

% MAE and MSE are calculated for each of the folds separately, then
% averaged across the folds. Let's do this by hand by first calculating the
% residuals:
residuals1 = y(1:2:end) - perf{3}{1};  % fold 1 with uneven trials
residuals2 = y(2:2:end) - perf{3}{2};  % fold 2 with even trials

% To calculate MAE, we need to average the absolute value of the residuals
% in each of the two folds first
mae1 = mean(abs(residuals1));   % MAE in fold 1
mae2 = mean(abs(residuals2));   % MAE in fold 2
% Since the folds can have different numbers of trials, weighted averages are
% used in MVPA Light wherein each fold is weighted by the number of samples
% it contains
N1 = numel(residuals1);
N2 = numel(residuals2);
mae = (mae1 * N1 + mae2 * N2) / (N1+N2); % weighted average 

% Comparing mae and perf{1} we see that they are indeed identical
[mae perf{1}]

% For MSE, the procedure is essentially the same, we only square the
% residuals before averaging:
mse1 = mean(residuals1 .^ 2);   % MSE in fold 1
mse2 = mean(residuals2 .^ 2);   % MSE in fold 2
mse = (mse1 * N1 + mse2 * N2) / (N1+N2); % weighted average 

% Comparing MSE to perf{2} again we see they are identical
[mse perf{2}]
