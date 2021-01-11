% unit test of mv_stat_wilcoxon_ranksum

rng(42)
tol = 10e-10;

% example X with duplicates for debugging:
% X = [10 -4 ; 1 10; 40 4; 10, -20; 0 -2; -30 20; 40 2; -10 10]

%% U-statistic using example from https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_nonparametric/BS704_Nonparametric4.html
placebo = [7.1 5 6 4 12]';
new_drug = [3 6 4 2 1]';
group = [1 1 1 1 1 2 2 2 2 2]';

[~, ~, U] = mv_stat_wilcoxon_ranksum([placebo; new_drug], group);

print_unittest_result('[U-statistic] handcoded example', 3, U, tol);

[zval, p, U] = mv_stat_wilcoxon_ranksum([placebo placebo; new_drug new_drug], group);

print_unittest_result('[U-statistic] handcoded example 2D', [3;3], U, tol);

%% check whether yields the same U-statistic as matlab's ranksum
n1s = [30, 55, 66, 80];
n2s = [10, 45, 99];

for n1 = n1s
    n1_label = ones(n1,1);
    for n2 = n2s
        X1 = randn(n1,1);
        X2 = randn(n2,1)+1;
        n2_label = 2*ones(n2,1);
        
        [~,~,stats] = ranksum(X1, X2, 'method','approximate');
        U_matlab =  stats.ranksum - n1*(n1+1)/2; % need to convert Wilcoxon ranksum into Mann-Whitney statistic
        [~, ~, U] = mv_stat_wilcoxon_ranksum([X1;X2], [n1_label; n2_label]);
        print_unittest_result(sprintf('[U-statistic] n1=%d, n2=%d compare to ranksum function', n1, n2), U_matlab, U, tol);
    end
end

%% check output for 3D data
sz = [4 5];
X = cat(1, randn([10 sz]), randn([11 sz])+1);
group = [ones(10,1); 2*ones(11,1)];

[zval, p, U] = mv_stat_wilcoxon_ranksum(X, group);
print_unittest_result('[zval] shape for 3D data', size(zval), sz, tol);
print_unittest_result('[p] shape for 3D data', size(p), sz, tol);
print_unittest_result('[p] all 0 <= p <= 1', all(abs(p(:))<=1), true, tol);
print_unittest_result('[U] shape for 3D data', size(U), sz, tol);

%% check output for 4D data
sz = [4 5 6];
X = cat(1, randn([10 sz]), randn([11 sz])+1);
group = [ones(10,1); 2*ones(11,1)];

[zval, p, U] = mv_stat_wilcoxon_ranksum(X, group);
print_unittest_result('[zval] shape for 3D data', size(zval), sz, tol);
print_unittest_result('[p] shape for 3D data', size(p), sz, tol);
print_unittest_result('[U] shape for 3D data', size(U), sz, tol);


