function X = mv_normalise(normalise, X)
% Normalises input data X. 
%
% Usage:
% X = mv_normalise(normalise,X)
%
%Parameters:
% normalise      - type of normalisation, 'zscore' 'demean'
% X              - [samples x features x time points] data matrix
%
%Returns:
% X              - normalised data

if isempty(normalise) || strcmp(normalise,'none')
    return
end

switch(normalise)
    case 'zscore'
        X = zscore(X,[],1);
        
    case 'demean'
        X  = X  - repmat(mean(X,1), [size(X,1) 1 1]);
        
    otherwise
        error('Unknown normalisation type: %s', normalise)
end