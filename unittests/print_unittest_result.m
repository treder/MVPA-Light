function pass = print_unittest_result(msg,expect, actual, tol)
% Prints the unittest result. Compares the expected result to the actual
% result. If the difference is < tol, the test is passed. Otherwise it is
% failed. 
%
% Parameters:
% msg           - [char] message string
% expect        - [double] expected result
% actual        - [double] actual result
% tol           - [double] tolerance. If |expect-actual| < tol we get PASS. 
%                 Otherwise a FAIL and a warning message is issued
global FAIL_COUNT

% make sure inputs are vectorized (because we cannot calculate norm on 3+
% dimensional arrays)
expect = expect(:);
actual = actual(:);

% Difference between expected and actual result
if numel(expect) == 1
    % input is scalar
    d = abs(expect-actual);
else
    % vector, matrix, or higher-dimensional array
    d = norm(expect-actual);
end

% turn expected and actual value into string for printing
expect_str = mat2str(expect);
actual_str = mat2str(actual);

maxlen = 20;
if numel(expect_str)>maxlen
    expect_str = [expect_str(1:maxlen), ' ...']; 
    if strcmp(expect_str(1),'['), expect_str = [expect_str ']']; end
end
if numel(actual_str)>maxlen
    actual_str = [actual_str(1:maxlen), ' ...']; 
    if strcmp(actual_str(1),'['), actual_str = [actual_str ']']; end
end

% result string
result = sprintf('%s: Expect = %s; Actual = %s; Diff = %d', msg, expect_str, actual_str, d);

if d < tol      % --- PASS ---
    fprintf('%s < %d', result, tol)
    fprintf('\tPASS\n')
    pass = 1;
elseif all(isnan(expect(:))) && all(isnan(actual(:))) % --- PASS ---
    fprintf('%s (both NaN)', result)
    fprintf('\tPASS\n')
    pass = 1;
else            % --- FAIL ---
    fprintf('%s > %d', result, tol)
    fprintf('\t[\b---- FAIL ----]\b\n')
    pass = 0;
    
    FAIL_COUNT = FAIL_COUNT + 1;
end

