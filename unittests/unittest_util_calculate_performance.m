% util unit test
%
% mv_calculate_performance 

% check the different metrics with predefined clabel and cf outputs
tol = 10e-10;
mf = mfilename;

% mv_calculate_performance(metric, output_type, cf_output, clabel, dim)

% number of instances in classes 1, 2 and 3
N1 = 100;
N2 = 200;
N3 = 300;

%% --- CLASSIFICATION METRICS --

%% ACCURACY
%% using class labels: check accuracy when it should be 0, 0.5, and 1 [two-class]
metric = 'accuracy';
output_type = 'clabel';
clabel = [ones(N1,1); 2*ones(N2,1)];

cf_output1 = clabel;    % accuracy = 1
cf_output0 = 3-clabel;  % accuracy = 0
cf_output05 = clabel;
cf_output05(1:N1/2) = 2;
cf_output05(N1+1:N1+N2/2) = 1;

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

print_unittest_result('[two-class accuracy] clabel completely different from cf_output', 0, perf0, tol);
print_unittest_result('[two-class accuracy] clabel identical to cf_output', 1, perf1, tol);
print_unittest_result('[two-class accuracy] clabel half identical to cf_output', 0.5, perf05, tol);

%% using class labels: check accuracy when it should be 0, 0.5, and 1 [multi-class]
metric = 'accuracy';
output_type = 'clabel';
clabel = [ones(N1,1); 2*ones(N2,1); 3*ones(N3,1)];

cf_output1 = clabel;                  % accuracy = 1
cf_output0 = mod(clabel + 1, 3) + 1;  % accuracy = 0
cf_output05 = clabel;
cf_output05(1:N1/2) = 2;
cf_output05(N1+1:N1+N2/2) = 3;
cf_output05(end-N3/2+1:end) = 1;

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

print_unittest_result('[multi-class accuracy] clabel completely different from cf_output', 0, perf0, tol);
print_unittest_result('[multi-class accuracy] clabel identical to cf_output', 1, perf1, tol);
print_unittest_result('[multi-class accuracy] clabel half identical to cf_output', 0.5, perf05, tol);

%% using dvals: check accuracy when it should be 0, 0.5, and 1 [two-class]
metric = 'accuracy';
output_type = 'dval';
clabel = [ones(N1,1); 2*ones(N2,1)];
dval = [abs(randn(N1,1)); -abs(randn(N2,1))];

cf_output1 = dval;    % accuracy = 1
cf_output0 = -dval;   % flip sign to get accuracy = 0
cf_output05 = dval;
cf_output05(1:N1/2) = -cf_output05(1:N1/2);  % flip sign of half of the sample
cf_output05(N1+1:N1+N2/2) = -cf_output05(N1+1:N1+N2/2);

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

print_unittest_result('[two-class accuracy] cf_output dvals have correct sign', 0, perf0, tol);
print_unittest_result('[two-class accuracy] cf_output dvals have opposite sign', 1, perf1, tol);
print_unittest_result('[two-class accuracy] half of cf_outputs dvals has correct sign', 0.5, perf05, tol);

%% using probabilities: check accuracy when it should be 0, 0.5, and 1 [two-class]
metric = 'accuracy';
output_type = 'prob';
clabel = [ones(N1,1); 2*ones(N2,1)];

% Generate probabilities from 0.5-1 for class 1 and 0-0.5 for class 2
p1 = randn(N1,1)/10+0.75;
p2 = randn(N2,1)/10+0.25;
prob = [min(max(p1,0.51),1); min(max(p2,0),0.49)];

cf_output1 = prob;    % accuracy = 1
cf_output0 = 1-prob;   % flip sign to get accuracy = 0
cf_output05 = prob;
cf_output05(1:N1/2) = 1-cf_output05(1:N1/2);  % flip sign of half of the sample
cf_output05(N1+1:N1+N2/2) = 1-cf_output05(N1+1:N1+N2/2);

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

print_unittest_result('[two-class accuracy] cf_output probabilities all wrong', 0, perf0, tol);
print_unittest_result('[two-class accuracy] cf_output probabilities all correct', 1, perf1, tol);
print_unittest_result('[two-class accuracy] cf_output probabilities half correct', 0.5, perf05, tol);

%% AUC
metric = 'auc';
output_type = 'dval';
clabel = [ones(N1,1); 2*ones(N2,1)];
dval = [abs(randn(N1,1)); -abs(randn(N2,1))];

cf_output1 = dval;
cf_output0 = -dval;

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);

print_unittest_result('[AUC] all dvals wrong sign', 0, perf0, tol);
print_unittest_result('[AUC] all dvals correct sign', 1, perf1, tol);


%% CONFUSION MATRIX
metric = 'confusion';
output_type = 'clabel';
clabel = [ones(N1,1); 2*ones(N2,1)];

cf_output0 = 3-clabel;
cf_output1 = clabel;  
cf_output05 = clabel;
cf_output05(1:N1/2) = 2;
cf_output05(N1+1:N1+N2/2) = 1;

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

c0 = [0 1; 1 0];
c1 = [1 0; 0 1];
c05 = [0.5 0.5; 0.5 0.5];

print_unittest_result('[confusion] matrix diff when clabel completely different from cf_output', 0, norm(perf0-c0), tol);
print_unittest_result('[confusion] matrix diff when clabel identical to cf_output', 0, norm(perf1-c1), tol);
print_unittest_result('[confusion] matrix diff when clabel half identical to cf_output', 0, norm(perf05-c05), tol);

%% DVAL
metric = 'dval';
output_type = 'dval';
clabel = [ones(N1,1); 2*ones(N2,1)];
d1 = randn(N1,1);
d2 = randn(N2,1);

% shift d1 to mean=-1 and d2 to mean=2
d1 = d1 - mean(d1) - 1;
d2 = d2 - mean(d2) + 2;
dval = [d1; d2];
cf_output = dval;

perf = mv_calculate_performance(metric, output_type, cf_output, clabel);

print_unittest_result('[dval] mean dval for class 1 is -1', -1, perf(1), tol);
print_unittest_result('[dval] mean dval for class 2 is 2', 2, perf(2), tol);

%% KAPPA

tol = 10e-5;

% Use the numerical examples from the Wikipedia page on Cohen's Kappa as a
% reference (https://en.wikipedia.org/wiki/Cohen's_kappa)
metric = 'kappa';
output_type = 'clabel';

% "Simple example" (https://en.wikipedia.org/wiki/Cohen's_kappa)
% Our contingency table needs to be
% 20  5 
% 10 15
% hence we need 50 class labels

clabel = [ones(25,1); 2*ones(25,1)];
cf_output = clabel;
cf_output(1:5) = 2;     % this gives us the 5
cf_output(26:35) = 1;   % this gives us the 10

perf = mv_calculate_performance(metric, output_type, cf_output, clabel);
print_unittest_result('[kappa] example 1, kappa should be 0.4', 0.4, perf(1), tol);

% Example "Same percentages but different numbers" (https://en.wikipedia.org/wiki/Cohen's_kappa)
% contingency table:
% 45 15 
% 25 15
% hence we need 100 class labels
clabel = [ones(60,1); 2*ones(40,1)];
cf_output = clabel;
cf_output(1:15) = 2;     % this gives us the 15
cf_output(61:85) = 1;    % this gives us the 25

perf = mv_calculate_performance(metric, output_type, cf_output, clabel);
print_unittest_result('[kappa] example 1, kappa should be 0.1304', 0.1304, perf(1), tol);

% contingency table:
% 25 35 
%  5 35
% hence we need 100 class labels
clabel = [ones(60,1); 2*ones(40,1)];
cf_output = clabel;
cf_output(1:35) = 2;     % this gives us the 35
cf_output(61:65) = 1;    % this gives us the  5

perf = mv_calculate_performance(metric, output_type, cf_output, clabel);
print_unittest_result('[kappa] example 1, kappa should be 0.2593', 0.2593, perf(1), tol);

%% TVAL
metric = 'tval';
output_type = 'dval';
clabel = [ones(N1,1); 2*ones(N2,1)];
d1 = randn(N1,1);
d2 = randn(N2,1);

% shift d1 to mean=-0.5 and d2 to mean=0.5
d1 = d1 - mean(d1) - 1;
d2 = d2 - mean(d2) + 1;
dval = [d1; d2];
cf_output = dval;

% perform t-test by hand (unequal sample size, equal variance)
[~,~,~,stat] = ttest2(d1,d2);

% calculate t-test using mv_calculate_performance
perf = mv_calculate_performance(metric, output_type, cf_output, clabel);

print_unittest_result('[tval] ttest2 result is equal to metric', stat.tstat, perf, tol);

%% PRECISION
metric = 'precision';
output_type = 'clabel';
clabel = [ones(N1,1); 2*ones(N2,1)];

cf_output0 = 3-clabel;
cf_output1 = clabel;  
cf_output05 = clabel;
cf_output05(1:N1/2) = 2;
cf_output05(N1+1:N1+N2/2) = 1;

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

print_unittest_result('[precision] when clabel do not match', 0, perf0, tol);
print_unittest_result('[precision] when clabel match perfectly', 1, perf1, tol);
print_unittest_result('[precision] when clabel half match', N1/(N1+N2), perf05, tol);

%% RECALL (using same outputs as in precision)
metric = 'recall';
output_type = 'clabel';

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

print_unittest_result('[recall] when clabel do not match', 0, perf0, tol);
print_unittest_result('[recall] when clabel match perfectly', 1, perf1, tol);
print_unittest_result('[recall] when clabel half match', 0.5, perf05, tol);

%% F1 score (using same outputs as in precision/recall)
metric = 'f1';
output_type = 'clabel';

perf0 = mv_calculate_performance(metric, output_type, cf_output0, clabel);
perf1 = mv_calculate_performance(metric, output_type, cf_output1, clabel);
perf05 = mv_calculate_performance(metric, output_type, cf_output05, clabel);

print_unittest_result('[f1] when clabel do not match', nan, perf0, tol);
print_unittest_result('[f1] when clabel match perfectly', 1, perf1, tol);
print_unittest_result('[f1] when clabel half match', N1/(N1+N2)/(N1/(N1+N2)+0.5), perf05, tol);

%% --- REGRESSION METRICS --

%% MSE / mean squared error
metric = 'mse';
y = abs(randn(N1, 3)*10);
y_hat = y; 

perf0 = mv_calculate_performance(metric, '', y_hat, y);
perf1 = mv_calculate_performance(metric, '', y_hat+2, y);

print_unittest_result('[MSE] when y_hat = y', 0, perf0, tol);
print_unittest_result('[MSE] when y_hat = y+2', 4, perf1, tol);

%% MAE / mean absolute error
metric = 'mae';
y = abs(randn(N1, 1)*10);
y_hat = y; 

perf0 = mv_calculate_performance(metric, '', y_hat, y);
perf1 = mv_calculate_performance(metric, '', y_hat+2, y);

print_unittest_result('[MAE] when y_hat = y', 0, perf0, tol);
print_unittest_result('[MAE] when y_hat = y+2', 2, perf1, tol);

%% R-squared
metric = 'r_squared';
y = abs(randn(N1, 1)*10);
y_hat0 = y; 

% regression relation
x_reg = randn(N1,1);
y_reg = x_reg + randn(N1,1);
B = regress(y_reg, x_reg);
y_hat1 = x_reg*B;

perf0 = mv_calculate_performance(metric, '', y_hat0, y);
perf1 = mv_calculate_performance(metric, '', y_hat1, y_reg);

% Let's invert y_hat1 which is equivalent to taking the negative slope: the
% R2 coefficient should now get negative since the model does not explain
% any variance but adds in variance
perf2 = mv_calculate_performance(metric, '', -y_hat1, y_reg);

print_unittest_result('[R-squared] when y_hat = y', 1, perf0,  tol);
print_unittest_result('[R-squared] between [0,1] after linear regression', 1, 0<=perf1<=1, tol);
print_unittest_result('[R-squared] negative when taking the negative slope', 1, perf2<0, tol);