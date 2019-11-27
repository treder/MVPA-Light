function lambda = LedoitWolfEstimate(X, form)
% Calculates the Ledoit-Wolf estimate for the shrinkage parameter used in
% the estimation of covariance matrices.
%
% Usage: lambda = LedoitWolfEstimate(X)
%
% X            - [samples x features] matrix of training samples
% form         - 'primal' uses the standard formula based on the covariance
%                matrix. The 'dual' form uses an alternate formula based on
%                the Gram matrix which is more efficient when the number of
%                features is larger than the number of samples (see
%                MVPA-Light paper for details)
%
% References:
% Ledoit, O., & Wolf, M. (2004).
% A well-conditioned estimator for large-dimensional covariance matrices. 
% Journal of Multivariate Analysis, 88(2), 365?411. 
% https://doi.org/10.1016/S0047-259X(03)00096-4

% (c) Matthias Treder

[n, p] = size(X);

% remove mean
X = X - repmat(mean(X,1), n, 1);

if strcmp(form, 'primal')
    % calculate lambda using covariance matrix
    
    % sample covariance matrix
    S = X'*X/n;
    
    X2 = X.^2;
    phi = sum(sum(X2'*X2/n - S.^2));
    gamma=norm(S - trace(S)*eye(p)/p,'fro')^2;
    
    % compute shrinkage constant
    lambda=(phi/gamma)/n;
    
  
    % Y.Chen formula:
    % SUM_i=1^n  || xi xi' - S||_F^2
%     numer = 0;
%     for ii=1:n
%         numer = numer +  norm(X(ii,:)' * X(ii,:) - S,'fro')^2;
%     end
%     denom = n^2 * (trace(S^2) - trace(S)^2 / p);
%     lambda = numer/denom;
    
elseif strcmp(form, 'dual')
    % calculate lambda using Gram matrix

    % Gram matrix
    G = X*X';
    
    traceG2 = trace(G^2);
    
    % numerator: rewrite the Chen formula SUM_i ||xi xi' - S||_F^2, this yields
    numer = traceG2/n + sum(diag(G).^2) - 2 * sum(sum(G.^2))/n;
    
    % denominator: same as for primal just replace S by G
    denom = traceG2 - trace(G)^2 / p;
    lambda = numer/denom;

    % previous version: slightly rewritten
    %     traceG2 = trace((G/n)^2);
%     
%     % numerator: rewrite the Chen formula SUM_i ||xi xi' - S||_F^2, this yields
%     numer = n * traceG2 + sum(diag(G).^2) - 2 * sum(sum(G.^2))/n;
%     
%     % denominator: same as for primal just replace S by G
%     denom = n^2 * (traceG2 - trace(G/n)^2 / p);
%     lambda = numer/denom;

end

% make sure there's no over/undershoot
lambda = max(0, min(lambda, 1));

