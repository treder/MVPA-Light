function CV = mv_get_crossvalidation_folds(cv, clabel, K, stratify, frac)
% Defines a cross-validation scheme and returns a cvpartition object with
% the definition of the folds.
%
% Usage:
% CV = mv_get_crossvalidation_folds(cv, clabel, K, stratify, P)
%
%Parameters:
% cv          - cross-validation type:
%               'kfold': K-fold cross-validation. The parameter K specifies
%               the number of folds.
%               'leave1out': leave-one-out cross-validation
% clabel      - vector of class labels
% K           - number of folds (the K in K-fold) (default 5)
% stratify    - if 1, class proportions are roughly preserved in
%               each fold (default 0)
% frac        - if cv_type is 'holdout', frac is the fraction of test samples
%                 (default 0.1)
%
%Output:
% CV - struct with cross-validation folds

% (c) Matthias Treder 2017


N = numel(clabel);

if nargin < 3,      K = 5; end
if nargin < 4,      stratify = 0; end
if nargin < 5,      frac = 0.1; end


switch(cv)
    case 'kfold'
        if stratify
            CV= cvpartition(clabel,'kfold', K);
        else
            CV= cvpartition(N, 'kfold', K);
        end
        
    case 'leaveout'
        CV= cvpartition(N,'leaveout');
        
    case 'holdout'
        if stratify
            CV= cvpartition(clabel,'holdout',frac);
        else
            CV= cvpartition(N,'holdout',frac);
        end
        
    otherwise error('Unknown cross-validation type: %s',cv)
end













% 
% % Leave-one-out is equal to k-fold when we set K = N
% if strcmp(cv_type,'leave1out')
%     K = N;
% end
% 
% %% Set up cross-validation struct
% CV = struct();
% CV.type = cv_type;
% CV.test = cell(K,1);
% CV.training = cell(K,1);
% 
% %% Get class frequencies
% nSamPerClass = arrayfun( @(u) sum(clabel == u), unique(clabel));
% freq = nSamPerClass / N;
% 
% %% Define folds
% switch(cv_type)
%     
%     %% --- K-FOLD ---
%     case {'kfold','leave1out'}
%         
%         % In order to stratify, we need to have more samples than folds in
%         % each class
%         if stratify && any(nSamPerClass < K)
%             warning('Some folds do not have instances of each class because there is too few')
%         end
%         if K==N
%             stratify = 0;
%         end
%         
%         % Randomly define folds
%         if stratify
%             
%         else
%         end
% 
%     otherwise error('Unknown cross-validation type: %s',cv_type)
% end
% 
