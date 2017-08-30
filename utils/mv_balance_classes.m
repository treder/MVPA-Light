function [X,label,labelidx] = mv_balance_classes(X,label,method,replace)
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
% method         - 'oversample' or 'undersample'. Alternatively, an integer
%                  number can be given which will be the number of samples
%                  in each class. Undersampling or oversampling of each
%                  class will be used to create it.
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
% X, label    - updated X and labels (oversampled samples appended)
% labelidx    - if labels are undersampled, labelidx gives the
%               indices/positions of the subset in the original
%               (bigger) label vector

% (c) Matthias Treder 2017

if nargin<4 || isempty(replace)
    replace = 1;
end

N = [sum(label==1), sum(label== -1)];
% N1 = sum(labels==1);
% N2 = sum(labels== -1);

%% Determine which class is the minority class
if N(1) < N(2)
    minorityClass = 1;
else
    minorityClass = -1;
end

%% Oversample/undersample
addRmSamples = abs(N(1)-N(2));  % number of samples to be added/removed for over/undersampling
labelidx = 1:sum(N);

if ischar(method) && strcmp(method,'oversample')
    % oversample the minority class
    idxMinority = find(label == minorityClass);
    if replace
        idxAdd = randi( min(N(1),N(2)), addRmSamples, 1);
    else
        idxAdd = randperm( min(N(1),N(2)), addRmSamples);
    end
    X= cat(1,X, X(idxMinority(idxAdd),:,:));
    label(end+1:end+addRmSamples)= label(idxMinority(idxAdd));
    
elseif ischar(method) && strcmp(method,'undersample')
    % undersample the majority class
    idxMajority = find(label == -1*minorityClass);
    idxRm = randperm( max(N(1),N(2)), addRmSamples);
    X(idxMajority(idxRm),:,:)= [];
    label(idxMajority(idxRm))= [];
    labelidx(idxMajority(idxRm))= [];
    
elseif isnumeric(method)
    % The target number of samples is directly provided as a number. Each
    % class will be either over- or undersampled to provide the target
    % number of trials. For instance, if N = [10 20] and num = 5, then both
    % classes will be undersampled such that there is 5 samples in each
    % class. If num=15, then the first class will be oversampled (adding 5
    % samples) and the second class will be undersampled (removing 5
    % samples). If num=30, both classes will be oversampled.
    num = method;
    
    % First check whether we need to over- or undersample each class
    do_undersample = [N(1) N(2)] > num;

    for cc=1:2
        y = (-1)^(cc+1);  % -1 for ii=1 and 1 for cc=2
        idxClass= find(label == y); % class indices for class cc
        
        if do_undersample(cc)
            % We need to undersample this class
            idxRm = randperm( N(cc), N(cc)-num );
            X(idxClass(idxRm),:,:)= [];
            label(idxClass(idxRm))= [];
            labelidx(idxClass(idxRm))= [];
        else
            % We need to oversample this class
            if num-N(cc)>0
                if replace
                    idxAdd = randi( N(cc), num-N(cc), 1);
                else
                    idxAdd = randperm( N(cc), num-N(cc));
                end
                X= cat(1,X, X(idxClass(idxAdd),:,:));
                label(end+1:end+num-N(cc))= label(idxClass(idxAdd));
            end
        end
    end
end

