function X = squeeze1(X)
% Like squeeze but always preserves the 1st dimension. This deals with
% cases where e.g. size(X) = [1, 1, 20] and size(squeeze(X)) = [20, 1]
% whereas size(squeeze(X)) = [1, 20] is desirable. The latter case is
% relevant e.g. in leaveout cross-validation where the first dimension is
% the sample dimension and the test set contains only 1 sample.
sz = size(X);
sz_end = sz(2:end);

% remove singleton dimensions except for the first dimension
sz = [sz(1) sz_end(~ismember(sz_end,1))];

if length(sz) == 1
    sz = [sz 1];
end

X = reshape(X, sz);