% Preprocessing unit test
%
% ssd
tol = 10e-10;

% Generate oscillatory data
cfg = [];
cfg.n_sample = 50;
cfg.n_channel = 16;
cfg.n_time_point = 512;
cfg.fs = 256;
cfg.n_narrow = 3;
cfg.freq = [1 2; 4 8; 8 12];
cfg.amplitude = [10 13 11];
cfg.narrow_class = [1 0; 0 1; 1 1];

[X, clabel] = simulate_oscillatory_data(cfg);
sz = size(X);

% 3D -> 2D then bandpass filter
Xtmp = reshape(permute(X, [3 1 2]), [], sz(2));
X_signal = bandpass(Xtmp, [8 12], cfg.fs);
X_noise = bandpass(Xtmp, [4 7], cfg.fs) + bandpass(Xtmp, [13 16], cfg.fs);

% 2D -> 3D
X_signal = permute(reshape(X_signal, sz(3), sz(1), sz(2)), [2 3 1]);
X_noise = permute(reshape(X_noise, sz(3), sz(1), sz(2)), [2 3 1]);

%% 3D example - check whether sizes are correct
% Get default parameters
n = 1;
pparam = mv_get_preprocess_param('ssd');
pparam.n = n;
pparam.target_dimension = 3;
pparam.feature_dimension = 2;
pparam.signal_train = X_signal;
pparam.noise_train = X_noise;
[~, Xout] = mv_preprocess_ssd(pparam, X);
print_unittest_result('size for 3D data with n=1', [sz(1), n, sz(3)], size(Xout), tol);

n = 2;
pparam.n = n;
pparam.target_dimension = 2;
pparam.feature_dimension = 1;
[~, Xout] = mv_preprocess_ssd(pparam, X);
print_unittest_result('size for 3D data (transposed) with n=2', [n, sz(2:3)], size(Xout), tol);

n = 5;
pparam.n = n;
pparam.target_dimension = 1;
pparam.feature_dimension = 3;
[~, Xout] = mv_preprocess_ssd(pparam, X);
print_unittest_result('size for 3D data (transposed) with n=5', [sz(1:2), n], size(Xout), tol);

%% 4D example - check whether sizes are correct
X2 = repmat(X, [1 1 1 4]);
X2 = X2 + randn(size(X2))*0.1; 
sz2 = size(X2);

pparam.n = 3;
pparam.feature_dimension = 3; 
pparam.target_dimension = 4;
[~, Xout] = mv_preprocess_ssd(pparam, X2);

print_unittest_result('size for 4D data', [sz2(1:2), pparam.n, sz2(4)], size(Xout), tol);

%% check whether dimension removed pparam.calculate_variance = 1
pparam.n = 3;
pparam.feature_dimension = 2; 
pparam.target_dimension = 3;
pparam.calculate_variance = true;

[~, Xout] = mv_preprocess_ssd(pparam, X);

print_unittest_result('dim for calculate_variance=1',[sz(1), pparam.n], size(Xout), tol);

%% spatial patterns and weights should be same size
pparam.n = 3;
pparam.feature_dimension = 2; 
pparam.target_dimension = 3;
pparam.calculate_variance = false;
pparam.calculate_spatial_pattern = true;

[pparam, Xout] = mv_preprocess_ssd(pparam, X);

print_unittest_result('size(pattern)= size(weights)',size(pparam.spatial_pattern), size(pparam.W), tol);

%% nested preprocessing: check size of output (should be single number)
pparam = mv_get_preprocess_param('ssd');
pparam.n = 12;
pparam.feature_dimension = 2;
pparam.target_dimension = 3;
pparam.calculate_variance = true;
pparam.signal = X_signal;
pparam.noise = X_noise;

cfg = [];
cfg.preprocess = 'ssd';
cfg.preprocess_param = pparam;
cfg.feature_dimension = [2 3];
cfg.flatten_features = false;
cfg.feedback = 0;

acc = mv_classify(cfg, X, clabel);

print_unittest_result('SSD in cross-validation', 1, numel(acc), 10^-2);
