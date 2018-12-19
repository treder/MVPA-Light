% Perform hyperparameter tuning for SVM. To this end, a nested
% cross-validation is performed by splitting the training data into folds.
%
% Accuracy is calculated for each possible hyperparameter value; the value
% leading to the best overall performance is then selected.

% Init variables
CV = cvpartition(N,'KFold',cfg.k);
acc = zeros(numel(cfg.c),1);

if cfg.plot
    c = zeros(numel(cfg.c));
end

%% TODO : build tuning grid for hyperparameters

LABEL = (clabel * clabel');

% --- Start cross-validation loop ---
for hyp=1:cfg.k          % ---- loop across hyperparameters ----
    
    %% TODO : the range in the loop needs fixing and here we need to re-evaluate
    %% the kernel matrix using the sets of hyperparameters
    
    for ff=1:cfg.k          % ---- loop across cost c ----
                
        %%% TODO: consider implementing full regularisation path according to
        %%% Hastie et al
        
        train_idx = find(CV.training(ff));
        test_idx = find(CV.test(ff));
        
        % Training data
        Xtrain = X(train_idx,:);
        Xtest= X(test_idx,:);
        trainlabel = clabel(train_idx);
        
        % Kernel submatrix for training data
        Qtrain = Q_cl(train_idx,train_idx);
        ONEtrain = ones(CV.TrainSize(ff),1);
        
        % --- Loop through c's ---
        for ll=1:numel(cfg.c)
            
            % Solve the dual problem and obtain alpha
            [alpha,iter] = DualCoordinateDescent(Qtrain, cfg.c(ll), ONEtrain, cfg.tolerance, cfg.shrinkage_multiplier);
            
            support_vector_indices = find(alpha>0);
            
            % For convenience we save the product [alpha * class label] for the support vectors
            alpha_y = alpha(support_vector_indices) .* trainlabel(support_vector_indices);
            
            % Exploit the fact that we already pre-calculated the kernel matrix
            % for all samples. Simply extract the values corresponding to
            % the test samples and the support vectors in the training set.
            dval = kernel_matrix(test_idx, train_idx(support_vector_indices)) * alpha_y;
            
            % accuracy
            acc(ll) = acc(ll) + sum( clabel(CV.test(ff)) == double(sign(dval(:)))  );
        end
        
        if cfg.plot
            c = c + corr(alpha);
        end
    end
end

% Average performance
acc = acc / N;

% Best overall parameter
[~, best_c_idx] = max(acc);

% Diagnostic plots if requested
if cfg.plot
    
    % Plot cross-validated classification performance
    figure
    semilogx(cfg.c, acc)
    title([num2str(cfg.k) '-fold cross-validation performance'])
    hold all
    plot([cfg.c(best_c_idx), cfg.c(best_c_idx)],ylim,'r--'),plot(cfg.c(best_c_idx), acc(best_c_idx),'ro')
    xlabel('Lambda'),ylabel('Accuracy'),grid on
    
end