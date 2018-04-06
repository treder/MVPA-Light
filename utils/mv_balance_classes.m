function [X,clabel,labelidx] = mv_balance_classes(X,clabel,method,replace)
%Balances data with imbalanced classes by oversampling the minority class
%or undersampling the majority class.
%
%Usage:
% [X,labels] = mv_balance_classes(X,clabel,method,replace)
%
%Parameters:
% X              - [samples x features x time points] or
%                  [samples x features] data matrix.
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
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

nclasses = max(clabel);

% Sample count for each class
N = arrayfun( @(c) sum(clabel==c) , 1:nclasses);

%% Determine which class is the minority/majority class
[~, minorityClass] = min(N);
[~, majorityClass] = max(N);

nmax = max(N);

% if N(1) < N(2)
%     minorityClass = 1;
%     majorityClass = 2;
% else
%     minorityClass = 2;
%     majorityClass = 1;
% end

%% Oversample/undersample

labelidx = 1:sum(N);

if ischar(method) && strcmp(method,'oversample')
    % oversample all but the majority classes
    addRmSamples = abs(N - max(N));
    for cc=1:nclasses
        if addRmSamples(cc)>0
            idxMinority = find(clabel == cc);
            if replace
                idxAdd = randi( numel(idxMinority), addRmSamples(cc), 1);
            else
                idxAdd = randperm( numel(idxMinority), addRmSamples(cc));
            end
            X= cat(1,X, X(idxMinority(idxAdd),:,:));
            clabel(end+1:end+addRmSamples(cc))= clabel(idxMinority(idxAdd));
        end
    end
elseif ischar(method) && strcmp(method,'undersample')
    % undersample the majority class
    addRmSamples = abs(N - min(N));
    for cc=1:nclasses
        if addRmSamples(cc)>0
            idxMajority = find(clabel == cc);
            idxRm = randperm( numel(idxMajority), addRmSamples(cc));
            X(idxMajority(idxRm),:,:)= [];
            clabel(idxMajority(idxRm))= [];
            labelidx(idxMajority(idxRm))= [];
        end
    end
elseif isnumeric(method)
    % The target number of samples is directly provided as a number. Each
    % class will be either over- or undersampled to provide the target
    % number of trials. For instance, if N = [10 20] and num = 5, then both
    % classes will be undersampled such that there is 5 samples in each
    % class. If num=15, then the first class will be oversampled (adding 5
    % samples) and the second class will be undersampled (removing 5
    % samples). If num=30, both classes will be oversampled.
    num = method;
   
    for cc=1:nclasses
        idxClass= find(clabel == cc); % class indices for class cc
        
        if N(cc) > num
            % We need to undersample this class
            idxRm = randperm( N(cc), N(cc)-num );
            X(idxClass(idxRm),:,:)= [];
            clabel(idxClass(idxRm))= [];
            labelidx(idxClass(idxRm))= [];
        elseif  N(cc) < num
            % We need to oversample this class
            if num-N(cc)>0
                if replace
                    idxAdd = randi( N(cc), num-N(cc), 1);
                else
                    idxAdd = randperm( N(cc), num-N(cc));
                end
                X= cat(1,X, X(idxClass(idxAdd),:,:));
                clabel(end+1:end+num-N(cc))= clabel(idxClass(idxAdd));
            end
        end
    end
end

