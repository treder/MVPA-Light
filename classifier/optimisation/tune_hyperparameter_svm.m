% Perform hyperparameter tuning for SVM. To this end, a nested
% cross-validation is performed by splitting the training data into folds.
%
% Accuracy is calculated for each possible hyperparameter value; the value
% leading to the best overall performance is then selected.

% Init variables
CV = cvpartition(N,'KFold',cfg.K);
acc = zeros(numel(cfg.C),1);

if cfg.plot
    C = zeros(numel(cfg.C));
end

LABEL = (clabel * clabel');

% --- Start cross-validation loop ---
for ff=1:cfg.K
    
    %%% TODO: consider adding tuning loop for other hyperparameters like gamma
    
    %%% TODO: check whether there's a bug: classification performances
    %%% expected to be more different for different parameters.
    
    train_idx = find(CV.training(ff));
    test_idx = find(CV.test(ff));
    
    % Training data
    Xtrain = X(train_idx,:);
    Xtest= X(test_idx,:);
    
    % Kernel submatrix
    Qtrain = Q_cl(train_idx,train_idx);
    ONEtrain = ones(CV.TrainSize(ff),1);
    
    % --- Loop through C's ---
    for ll=1:numel(cfg.C)
        
        % Solve the dual problem and obtain alpha
        [alpha,iter] = DualCoordinateDescent(Qtrain, cfg.C(ll), ONEtrain, cfg.tolerance, cfg.shrinkage_multiplier);
        
        support_vector_indices = find(alpha>0);
        support_vectors  = Xtrain(support_vector_indices,:);
        
        % Class labels for the support vectors
        y                = clabel(support_vector_indices);
        
        % For convenience we save the product alpha * y for the support vectors
        alpha_y = alpha(support_vector_indices) .* y(:);
        
        % Exploit the fact that we already pre-calculated the kernel matrix
        % for all samples. Simply extract the values corresponding to
        % the test samples and the support vectors in the training set.
        dval = Q(test_idx, train_idx(support_vector_indices)) * alpha_y;
        
        % accuracy
        acc(ll) = acc(ll) + sum( clabel(CV.test(ff)) == double(sign(dval(:)))  );
    end
    
    if cfg.plot
        C = C + corr(alpha);
    end
end

% Average performance
acc = acc / N;

% Best overall parameter
[~, best_idx] = max(acc);

% Diagnostic plots if requested
if cfg.plot
    
    % Plot cross-validated classification performance
    figure
    semilogx(cfg.C,acc)
    title([num2str(cfg.K) '-fold cross-validation performance'])
    hold all
    plot([cfg.C(best_idx), cfg.C(best_idx)],ylim,'r--'),plot(cfg.C(best_idx), acc(best_idx),'ro')
    xlabel('Lambda'),ylabel('Accuracy'),grid on
    
    
end