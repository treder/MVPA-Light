% Hyperparameter tuning for SVM using nested cross-validation.
%
% Accuracy is calculated for each possible hyperparameter values; the
% values leading to the best overall performance are selected.

% Init variables
CV = cvpartition(N,'KFold', param.k);

if param.plot
    c = zeros(numel(param.c));
end

if isempty(tune_kernel_parameters)
    acc = zeros(numel(param.c), 1);
    kernel_comb = 1;
    ix = {};
else
    acc = zeros([cellfun(@numel, tune_kernel_parameters(2:2:end)), numel(param.c)]);
    LABEL = (clabel * clabel');
    
    % Create search grid for kernel parameters
    kernel_comb = allcomb(tune_kernel_parameters{2:2:end});
    ixcomb = cellfun(@(x) 1:numel(x), tune_kernel_parameters(2:2:end), 'Un', 0);
    kernel_comb_ix = allcomb(ixcomb{:});
end

% --- Start cross-validation loop ---

for kk=1:size(kernel_comb,1)       % -- kernel parameter search grid
    
    if ~isempty(tune_kernel_parameters)
        ix = {};
        % we tune the kernel parameters too, so need to recompute the
        % kernel matrix
        tmp_param = param;
        for jj=1:numel(tune_kernel_parameters)/2
            ix{jj} = kernel_comb_ix(kk,jj); % indices needed to access acc
            tmp_param.(tune_kernel_parameters{jj*2-1}) = kernel_comb(kk,jj);
        end
        
        % Compute kernel matrix
        K = kernelfun(tmp_param, X);
        
        % Regularize
        if param.regularize_kernel > 0
            K = K + param.regularize_kernel * eye(size(X,1));
        end
        
        Q = K .* LABEL;
    end
    
    for ff=1:param.k        % --- CV folds
        
        %%% TODO: consider implementing full regularisation path according to
        %%% Hastie et al
        
        train_idx = find(CV.training(ff));
        test_idx = find(CV.test(ff));
        
        trainlabel = clabel(train_idx);
        ONEtrain = ones(CV.TrainSize(ff),1);
        
        % --- Loop through c's ---
        for ll=1:numel(param.c)
            
            % Solve the dual problem and obtain alpha
            [alpha,iter] = DualCoordinateDescent(Q(train_idx, train_idx), param.c(ll), ONEtrain, param.tolerance, param.shrinkage_multiplier);
            
            support_vector_indices = find(alpha>0);
            
            % For convenience we save the product [alpha * class label] for the support vectors
            alpha_y = alpha(support_vector_indices) .* trainlabel(support_vector_indices);
            
            % Exploit the fact that we already pre-calculated the kernel matrix
            % for all samples. Simply extract the values corresponding to
            % the test samples and the support vectors in the training set.
            dval = K(test_idx, train_idx(support_vector_indices)) * alpha_y;
            
            % accuracy
            acc(ix{:}, ll) = acc(ix{:}, ll) + sum( clabel(CV.test(ff)) == double(sign(dval(:)))  );
        end
        
        if param.plot
            c = c + corr(alpha);
        end
    end
end

% Average performance
acc = acc / N;

% look for peak accuracy to determine best parameters
if isempty(tune_kernel_parameters)
    [~, ix] = max(acc);
else
    % find best parameters from multi-dimensional grid
    [~, ind] = max(acc(:));
    ix = cell(numel(tune_kernel_parameters)/2 + 1,1);
    [ix{:}] = ind2sub(size(acc), ind);
    ix = [ix{:}];
    
    % Best kernel parameters
    for jj=1:numel(tune_kernel_parameters)/2
        param.(tune_kernel_parameters{jj*2-1}) = tune_kernel_parameters{jj*2}(ix(jj));
    end

    % Compute best kernel matrix
    K = kernelfun(param, X);
    
    % Regularize
    if param.regularize_kernel > 0
        K = K + param.regularize_kernel * eye(size(X,1));
    end

    Q = K .* LABEL;
end

% Diagnostic plots if requested [only if kernel hyperparameters are not
% tuned]
if param.plot
    
    % Plot cross-validated classification performance
    figure
    semilogx(param.c, acc)
    title([num2str(param.k) '-fold cross-validation performance'])
    hold all
    plot([param.c(ix), param.c(ix)],ylim,'r--'),plot(param.c(ix), acc(ix),'ro')
    xlabel('Lambda'),ylabel('Accuracy'),grid on

end

% Select best c
param.c = param.c(ix(end));
