function U = mv_repelem(V, N)
%Drop-in replacement for Matlab's repelem function (which is not supported
%by some earlier versions of Matlab).
%
%Usage:
% U = repelem(V,N)
%
%Parameters:
% V       - [scalar or vector] elements to be repeated
% N       - [scalar of vector] number of times each element is repeated (if
%                              N scalar), otherwise N(i) is the number of
%                              times the i-th element is repeated (N
%                              vector)
%
%Returns:
% U - vector with replicated elements

n_V = numel(V);
n_N = numel(N);

if (n_V == 1) && (n_N == 1)
    U = repmat(V, [1 N]);
   
elseif (n_V>1) && n_N==1
    if size(V,1)==1  % V is row vector
        U = repmat(V, [N, 1]);
        U = U(:)';
    else  % V is column vector
        U = repmat(V, [1, N])';
        U = U(:);
    end
else
    U = arrayfun(@(v,c) repmat(v, [1 c]), V(:), N(:), 'Un', 0);
    if size(V,1)==1  % V is row vector
        U = [U{:}];
    else  % V is column vector
        U = [U{:}]';
    end
    
end
