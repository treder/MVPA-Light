function X = squeeze1(X)
% Like squeeze but prevents unwanted transposes. This deals with
% cases where e.g. size(X) = [1, 1, 20] and size(squeeze(X)) = [20, 1]
% whereas size(squeeze(X)) = [1, 20] would be desirable. The latter case is
% relevant e.g. in leaveout cross-validation where the first dimension is
% the sample dimension and the test set contains only 1 sample.
sz = size(X);

if sum(sz>1) > 1
    % there's more than 2 non-singleton dimensions: we can simply use
    % squeeze in this case
    X = squeeze(X);
else
    sz_end = sz(2:end);

    % remove singleton dimensions except for the first dimension
    sz = [sz(1) sz_end(~ismember(sz_end,1))];

    if length(sz) == 1
        sz = [sz 1];
    end

    X = reshape(X, sz);
end
