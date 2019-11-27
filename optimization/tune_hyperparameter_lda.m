% Inverting the covariance matrix is computationally expensive. To reduce
% computation time, PCA is performed first. The covariance matrix is then
% diagonal, which is easy to invert.

% Eigenvalue decomposition
X = bsxfun(@minus,X, mean(X));
[V,D] = eig(cov(X));
D= diag(D);

% Sort by descending eigenvalue
[D, soidx] = sort(D,'descend');
V= V(:,soidx);

% Reduce the data to well-conditioned subspace if necessary
idx =  find( ((D(1)./D) > cfg.evtol), 1);
if ~isempty(idx)
    D= D(1:idx);
    V= V(:,1:idx);
end

% Project data onto PCs
Xpca = X*V;

% Cross-validation
CV = cvpartition(N,'KFold',cfg.k);
ws = zeros(numel(D), numel(cfg.lambda));
acc = zeros(numel(cfg.lambda),1);

for ff=1:cfg.k
    Xtrain = Xpca(CV.training(ff),:);
    idx1 = clabel(CV.training(ff))==1;
    idx2 = clabel(CV.training(ff))==2;
    Ctrain= sum(idx1) * cov(Xtrain(idx1,:)) + sum(idx2) * cov(Xtrain(idx2,:));
    Ctrain= diag(Ctrain);
    
    % normalisation factor for regularisation target
    nu = sum(Ctrain)/numel(Ctrain); 
    
    % Class means and difference pattern
    mu1pca= mean(Xtrain(idx1,:))';
    mu2pca= mean(Xtrain(idx2,:))';
    p = mu1pca - mu2pca;
    testlabel = clabel(CV.test(ff));
    
    for ll=1:numel(cfg.lambda)
        ws(:,ll) = (1./ ((1-cfg.lambda(ll))*Ctrain + cfg.lambda(ll)*nu)) .* p;
    end
    
    % Threshold
    b= ws'*(mu1pca+mu2pca)/2;
    
    % Calculate decision values and predicted labels
    dval = bsxfun(@minus, Xpca(CV.test(ff),:) * ws, b');
    predlabel = double(dval >= 0) + 2*double(dval < 0);
    
    % Calculate accuracy
    acc = acc + sum(bsxfun(@eq, predlabel, testlabel))' / CV.TestSize(ff);
    
end

acc = acc / cfg.k;

[~, idx] = max(acc);
lambda = cfg.lambda(idx);

% Diagnostic plots if requested
if cfg.plot
    % Plot regularisation path (for the last training fold)
    figure
    for ii=1:numel(D), semilogx(cfg.lambda,ws(ii,:),'-'), hold all, end
    plot(xlim,[0,0],'k-'),title('Regularisation path for last iteration'),xlabel('Lambda')
    
    % Plot cross-validated classification performance
    figure
    semilogx(cfg.lambda, acc)
    title([num2str(cfg.k) '-fold cross-validation performance'])
    hold all
    plot([lambda, lambda],ylim,'r--'),plot(lambda, acc(idx),'ro')
    xlabel('Lambda'),ylabel('Accuracy')
end

acc= [];