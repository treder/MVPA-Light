% Loop through all lambdas

% Cross-validation
CV = cvpartition(N,'KFold',param.k);
MSE = zeros(numel(param.lambda),1);  % sum of squared errors

for ff=1:param.k
    X_train = X(CV.training(ff),:);
    Y_train = Y(CV.training(ff),:);
    
    m = mean(X_train);
    X_train = X_train - repmat(m, [size(X_train,1) 1]);

    % Loop through lambdas
    for ll=1:numel(param.lambda)
        lambda = param.lambda(ll);
        
        %%% TRAIN
        % Perform regularization and calculate weights
        if strcmp(form, 'primal')
            w = (X_train'*X_train + lambda * eye(P)) \ (X_train' * Y_train);   % primal
        else
            w = X_train' * ((X_train*X_train' + lambda * eye(N)) \ Y_train);   % dual
        end
        
        % Estimate intercept
        b = mean(Y_train) - m*w; % m*w makes sure that we do not need to center the test data
        
        %%% TEST
        Y_hat = X(CV.test(ff),:) * w + b;
        
        MSE(ll) = sum(sum( (Y(CV.test(ff),:) - Y_hat).^2 ));
    end
    
end

MSE = MSE / N;

[~, ix] = min(MSE);
lambda = param.lambda(ix);

% Diagnostic plot if requested
if param.plot

    % Plot cross-validated classification performance
    figure
    semilogx(param.lambda, MSE)
    title([num2str(param.k) '-fold cross-validation error'])
    hold all
    plot([lambda, lambda],ylim,'r--'),plot(lambda, MSE(ix),'ro')
    xlabel('Lambda'),ylabel('MSE')
end
