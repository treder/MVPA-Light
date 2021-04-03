% fix_preprocess_dimensions unit test
%

rng(42)
tol = 10e-10;

% Random data
N = 100;
X = randn(N,40);
clabel = randi(2, N, 1);

cfg = [];
cfg.preprocess = {};
cfg.preprocess_param = {};

oversample_param = mv_get_preprocess_param('oversample');
undersample_param = mv_get_preprocess_param('undersample');
zscore_param = mv_get_preprocess_param('zscore');
demean_param = mv_get_preprocess_param('demean');
average_param = mv_get_preprocess_param('average_samples');
pca_param = mv_get_preprocess_param('pca');


%% single pparam with dim_order = [3 2 1 4]
pca_param.feature_dimension = 2;
pca_param.target_dimension  = 3;

tmp = mv_rearrange_preprocess_dimensions(pca_param, [3 2 1 4]);

print_unittest_result('single pparam with dim_order=[3,2,1]', [2 1], [tmp.feature_dimension tmp.target_dimension], tol);

%% multiple pparam shift dim by 1 (dim_order=[5 1 2 3 4])
% put them all into preprocess parameters
pparam = {oversample_param undersample_param zscore_param demean_param average_param pca_param};
tmp = mv_rearrange_preprocess_dimensions(pparam, [5 1 2 3 4]);

p1 = [];
p2 = [];
for pp = 1:numel(pparam)
   fn = fieldnames(pparam{pp});
   dim_ix = find(arrayfun(@(x) ~isempty(x{1}), strfind(fn,'dimension')) );
   for ix = 1:numel(dim_ix)
       p1 = [p1 pparam{pp}.(fn{dim_ix(ix)})];
       p2 = [p2 tmp{pp}.(fn{dim_ix(ix)})];
   end
end

print_unittest_result('multiple pparam shift dim by 1 (dim_order=[5 1 2 3 4])', p1+1, p2, tol);

%% multiple pparam shift dim by 2
% put them all into preprocess parameters
pparam = {oversample_param undersample_param zscore_param demean_param average_param pca_param};
tmp = mv_rearrange_preprocess_dimensions(pparam, [4 5 1 2 3]);

p1 = [];
p2 = [];
for pp = 1:numel(pparam)
   fn = fieldnames(pparam{pp});
   dim_ix = find(arrayfun(@(x) ~isempty(x{1}), strfind(fn,'dimension')) );
   for ix = 1:numel(dim_ix)
       p1 = [p1 pparam{pp}.(fn{dim_ix(ix)})];
       p2 = [p2 tmp{pp}.(fn{dim_ix(ix)})];
   end
end

print_unittest_result('multiple pparam shift dim by 2', p1+2, p2, tol);

%% multiple pparam shift dim by 2 and max_dim = 3
max_dim = 3;
pparam = {oversample_param undersample_param zscore_param demean_param average_param pca_param};
tmp = mv_rearrange_preprocess_dimensions(pparam, [4 5 1 2 3], max_dim);

p1 = [];
p2 = [];
for pp = 1:numel(pparam)
   fn = fieldnames(pparam{pp});
   dim_ix = find(arrayfun(@(x) ~isempty(x{1}), strfind(fn,'dimension')) );
   for ix = 1:numel(dim_ix)
       p1 = [p1 pparam{pp}.(fn{dim_ix(ix)})];
       p2 = [p2 tmp{pp}.(fn{dim_ix(ix)})];
   end
end

print_unittest_result('multiple pparam shift dim by 1 (dim_order=[4 5 1 2 3])', min(p1+2, max_dim), p2, tol);





