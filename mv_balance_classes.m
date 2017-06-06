function [X,labels] = mv_balance_classes(X,labels,method,replace)
%Balances data with imbalanced classes by oversampling the minority class
%or undersampling the majority class.
%
%Usage:
% [X,labels] = mv_balance_classes(X,labels,method,replace)
%
%Parameters:
% X              - [number of samples x number of features x number of time points]
%                  or [number of samples x number of features] data matrix.
% labels         - [1 x number of samples] vector of class labels containing
%                  1's (class 1) and -1's (class 2)
% method         - 'oversample' or 'undersample'
% replace        - for oversampling, if 1, data is oversampled with replacement (like in
%                  bootstrapping) i.e. samples from the minority class can
%                  be added 3 or more times. If 0, data is oversampled
%                  without replacement. Note: If 0, the majority class must
%                  be at most twice as large as the minority class
%                  (otherwise we run out of samples) (default 1)
% 
%Note: If you use oversampling with cross-validation, the oversampling
%needs to be done *within* each training fold. The reason is that if the
%oversampling is done globally (before starting cross-validation), the
%training and test sets are not independent any more because they might
%contain identical samples (a sample could be in the training data and its
%copy in the test data). This will make life easier for the classifier and
%will lead to an artificially inflated performance.
%
%Undersampling can be done globally since it does not introduce any
%dependencies between samples. Since samples are randomly
%subselected in undersampling, a cross-validation should be repeated
%several times, each time with a fresh undersampled dataset.
%
%These two principles are implemented in mv_classify_across_time and
%my_classify_time-by-time
%
%Returns:
% X, labels    - updated X and labels (oversampled samples appended)

% (c) Matthias Treder 2017

if nargin<4 || isempty(replace)
    replace = 1;
end

N1 = sum(labels==1);
N2 = sum(labels== -1);

%% Determine which class is the minority class
if N1 < N2
    minorityClass = 1;
else
    minorityClass = -1;
end

%% Oversample/undersample
addRmSamples = abs(N1-N2);  % number of samples to be added/removed

if strcmp(method,'oversample')
    % oversample the minority class
    idxMinority = find(labels == minorityClass);
    if replace
        idxAdd = randi( min(N1,N2), addRmSamples, 1);
    else
        idxAdd = randperm( min(N1,N2), addRmSamples);
    end
    X= cat(1,X, X(idxMinority(idxAdd),:,:));
    labels(end+1:end+addRmSamples)= labels(idxMinority(idxAdd));
    
elseif strcmp(method,'undersample')
    % undersample the majority class
    idxMajority = find(labels == -1*minorityClass);
    idxRm = randperm( max(N1,N2), addRmSamples);
    X(idxMajority(idxRm),:,:)= [];
    labels(idxMajority(idxRm))= [];
end

