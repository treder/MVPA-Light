% Preprocessing unit test
%
% mmn
tol = 10e-10;

% Generate oscillatory data
cfg = [];
cfg.n_sample = 50;
cfg.n_channel = 16;
cfg.n_time_point = 80;
cfg.fs = 256;
cfg.n_narrow = 3;
cfg.freq = [1 2; 4 8; 8 12];
cfg.amplitude = [10 13 11];
cfg.narrow_class = [1 0; 0 1; 1 1];
cfg.narrow_weight = [];

[X, clabel] = simulate_oscillatory_data(cfg);
sz = size(X);

%% 2D example - check whether sizes are correct
pparam = mv_get_preprocess_param('mmn');
pparam.target_dimension = [];
X2 = X(:, :, 1);
[pparam_out, ~] = mv_preprocess_mmn(pparam, X2, clabel);
print_unittest_result('[X=2d] size of C_invsqrt', [size(X2, 2), size(X2,2)], size(pparam_out.C_invsqrt), tol);

%% 3D example - check whether sizes are correct
pparam = mv_get_preprocess_param('mmn');

pparam.target_dimension = [];
[pparam_out, ~] = mv_preprocess_mmn(pparam, X, clabel);
print_unittest_result('[X=3d] size of C_invsqrt with target_dimension=[]', [sz(3), sz(2), sz(2)], size(pparam_out.C_invsqrt), tol);

pparam.sample_dimension = 1;
pparam.feature_dimension = 2;
pparam.target_dimension = 3;
[pparam_out, ~] = mv_preprocess_mmn(pparam, X, clabel);
print_unittest_result('[X=3d] size of C_invsqrt with target_dimension=3', [size(X, 2), size(X, 2)], size(pparam_out.C_invsqrt), tol);

pparam.feature_dimension = 3;
pparam.target_dimension = 2;
[pparam_out, ~] = mv_preprocess_mmn(pparam, X, clabel);
print_unittest_result('[X=3d] size of C_invsqrt with target_dimension=3', [sz(3), sz(3)], size(pparam_out.C_invsqrt), tol);

%% 3D example - use target_indices (output should be same size)
pparam.sample_dimension = 1;
pparam.feature_dimension = 2;
pparam.target_dimension = 3;
pparam.target_indices = 1:15;
[pparam_out, ~] = mv_preprocess_mmn(pparam, X, clabel);
print_unittest_result('[X=3d] using target_indices', [size(X, 2), size(X, 2)], size(pparam_out.C_invsqrt), tol);

%% 4D example - check whether sizes are correct
X2 = repmat(X, [1 1 1 5]);
X2 = X2 + randn(size(X2))*0.1; 
sz = size(X2);

pparam = mv_get_preprocess_param('mmn');
pparam.target_dimension = [];
pparam.feature_dimension = 3;
[pparam_out, ~] = mv_preprocess_mmn(pparam, X2, clabel);

print_unittest_result('[X=4d] size of C_invsqrt with target_dimension=3', [sz(2), sz(4), sz(3), sz(3)], size(pparam_out.C_invsqrt), tol);

pparam = mv_get_preprocess_param('mmn');
pparam.target_dimension = 3;
[pparam_out, ~] = mv_preprocess_mmn(pparam, X2, clabel);

print_unittest_result('[X=4d] size of C_invsqrt with target_dimension=3', [sz(4), sz(2), sz(2)], size(pparam_out.C_invsqrt), tol);

%% 4D example - train and test
X2_train = X2(1:20,:,:,:,:);
X2_test = X2(21:end,:,:,:,:);
pparam = mv_get_preprocess_param('mmn');
pparam.target_dimension = 3;
[pparam_out, ~] = mv_preprocess_mmn(pparam, X2_train, clabel(1:20));
print_unittest_result('[X=4d] size of train C_invsqrt', [size(X2_train,4),size(X2_train,2),size(X2_train,2)], size(pparam_out.C_invsqrt), tol);

% test
pparam_out.is_train_set = 0;
[pparam_out, X2_out] = mv_preprocess_mmn(pparam_out, X2_test, clabel(21:end));
print_unittest_result('[X=4d] size of test output', size(X2_test), size(X2_out), tol);

