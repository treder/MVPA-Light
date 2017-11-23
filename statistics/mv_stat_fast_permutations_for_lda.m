function stat = mv_stat_fast_permutations_for_lda(cfg, X0, clabel)
% Fast permutations for regularised LDA.
%
% Usage:
% stat = mv_stat_fast_permutations_for_lda(cfg,X,clabel)
%
% X0             - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing
%                  1's (class 1) and 2's (class 2)
% num_per        - number of permutations (default 1000)
% regularisation - type of regularisation, 'none' for no regularisation.
%                 'ridge': ridge-type regularisation of C + lambda*I,
%                          where C is the covariance matrix and I is the
%                          identity matrix
%                 'shrink': shrinkage regularisation using (1-lambda)*C +
%                          lambda*nu*I, where nu = trace(C)/F and F =
%                          number of features. nu assures that the trace of
%                          C is equal to the trace of the regularisation
%                          term (default 'shrink')
% lambda         - value of the lambda parameter. If
%                  regularisation='shrink' and lambda='auto', the
%                  Ledoit-Wolf automatic estimation procedure is used to
%                  estimate lambda in each iteration. (default 'auto')
% rng            - if a number is provided, sets the random generator seed
%                  to the according value. Per default rng = [], so the
%                  seed is not being reset
% woodbury       - if 1, uses the Woodbury matrix identity for fast
%                  permutations. If 0, calculates the inverse directly
%                  (default 1)
% plot           - if 1, produces a bar plot of the accuracies obtained 
%                  with the permutations and marks the original accuracy
%
% For further cross-validation settings, see the parameter description in
% mv_crossvalidate.

[N,F] = size(X0);

mv_set_default(cfg,'num_per',1000);
mv_set_default(cfg,'regularisation','shrink');
mv_set_default(cfg,'lambda','auto');
mv_set_default(cfg,'rng',[]);
mv_set_default(cfg,'woodbury',1);
mv_set_default(cfg,'plot',1);


% Cross-validation settings
mv_set_default(cfg,'CV','kfold');
mv_set_default(cfg,'repeat',1);
mv_set_default(cfg,'K',10);
mv_set_default(cfg,'stratify',0);

I2 = eye(2);
num_per = cfg.num_per;

%% Accuracy
acc = zeros(cfg.repeat, cfg.num_per+1); % last element will be the original one

%% Set random number generator
if ~isempty(cfg.rng)
    rng(cfg.rng);
end

%% Run permutations
stat = [];


for rr=1:cfg.repeat                 % ---- CV repetitions ----
    
    CV = mv_get_crossvalidation_folds(cfg.CV, clabel, cfg.K, cfg.stratify);
    
    %% Prepare all permutations of the class labels
    plabel = zeros(N, num_per+1);
    for pp=1:num_per
        plabel(:,pp) = clabel(randperm(N));
    end
    % Attach matrix with true labels
    plabel(:,end) = clabel;
    % Create indicator matrix for fast access to class 1 and 2
    pind = cat(3, plabel==1, plabel==2);
    
    %% CV
    for kk=1:CV.NumTestSets                     % ---- CV folds ----
        
        % Training data, test data and real labels
        X = X0(CV.training(kk),:,:,:);
        Xtest = X0(CV.test(kk),:,:,:);
        
        %% Calculate and regularise scatter matrix
        S = X' * X;
        
        if strcmp(cfg.regularise,'shrink')
            error('todo: shrinkage')
        elseif strcmp(cfg.regularise,'ridge')
            S = S + cfg.lambda * eye(F);
        end
        
        
        %% Calculate inverse (regularised) scatter matrix
        Sinv = inv(S);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%  PERFORMANCE USING TRUE LABELS  %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        testlabel_true = clabel(CV.test(kk));

        %% Calculate class frequencies in training data
        N1 = sum(pind(CV.training(kk),end,1));
        N2 = sum(pind(CV.training(kk),end,2));

        %% Means and grand mean
        m1 = sum(X(pind(CV.training(kk),end,1),:))';
        m2 = sum(X(pind(CV.training(kk),end,2),:))';
        m = [m1/sqrt(N1), m2/sqrt(N2)];

        %% Calculate w and b
        w = (S - m*m')\(m1-m2);
        
        % Calculate threshold b = center between both classes
        b = w' * (m1 + m2)/2;
        
        %% calculate accuracy for the true class labels
        if cfg.woodbury
            Sinvm= Sinv*m;
            Cinv = Sinv + Sinvm/inv(I2 - m'*Sinvm) * Sinvm';
        else
            Cinv = inv(S - m*m');
        end
        
        %% Predicted class labels on test set
        dval = Xtest*w - b;
        predlabel= double(dval >= 0) + 2*double(dval < 0);
        
        %% Calculate accuracy on test set
        acc(num_per+1) = acc(num_per+1) + sum(predlabel(:) == testlabel_true(:));
        
        if cfg.woodbury
            % Pre-compute matrix product for inversion lemma
            SinvX = Sinv*X';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%        PERMUTATIONS        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for pp=1:num_per             

            % Get train labels for permuted data
            trainlabel_perm = plabel(CV.training(kk),pp);

            %% Calculate class frequencies in permuted training data
            N1 = sum(pind(CV.training(kk),pp,1));
            N2 = sum(pind(CV.training(kk),pp,2));
            
            %% Calculate new means
            m1 = sum(X(pind(CV.training(kk),pp,1),:))';
            m2 = sum(X(pind(CV.training(kk),pp,2),:))';
            
            %% Update the inverse scatter matrix
            if cfg.woodbury
                m = [m1/sqrt(N1), m2/sqrt(N2)];
                % Use the Woodbury matrix identity (fast way):
                % We want to calculate (Sinv + (-m)*m')^-1, by the Woodbury
                % matrix identity this is given by
                Sinvm = Sinv*m;
%                 Sinvm = [sum(SinvX(:,pind(CV.training(kk),pp,1)),2)/sqrt(N1), ...
%                     sum(SinvX(:,pind(CV.training(kk),pp,2)),2)/sqrt(N2)];
                Cinv = Sinv + Sinvm/(I2 - m'*Sinvm) * Sinvm';
            
            else
                % control: calculate Sinv_perm directly the slow way to compare
                % Cinv = inv(N1*cov(X(plabel==1,:),1)  + N2*cov(X(plabel==2,:),1));
                % or
                Cinv = inv(S - m*m');
%                 Cinv = inv(N1 * cov(X(pind(CV.training(kk),pp,1),:),1) + ...
%                     N2 * cov(X(pind(CV.training(kk),pp,2),:),1) );
            end
            
            %% Calculate w
            w = Cinv * (m1-m2); 
            
            %% Update threshold b (only necessary if N1 and N2 are not equal)
            if N1 ~= N2
                b = w' * (m1 + m2)/2;
            end
            
            %% Predicted class labels on test set
            dval = Xtest*w - b;
            predlabel= double(dval >= 0) + 2*double(dval < 0);
            
            %% Calculate accuracy on test set
            acc(pp) = acc(pp) + sum(predlabel(:) == plabel(CV.test(kk),pp));
            
        end
        
        
        
    end
end

% Calculate accuracy by dividing by the number of samples x repeats
acc = acc / N / cfg.repeat;

% Set output arguments
stat.acc = acc;
stat.p = 1 - sum(acc(end) > acc(1:end-1))/num_per;

% Bar plot
if cfg.plot
    clf
    hist(acc,200)
end