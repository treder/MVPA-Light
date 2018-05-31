% function pass = unittest_print_result(caller, msg,expect, actual, tol)
function pass = unittest_print_result(msg,expect, actual, tol)
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

% Difference between expected and actual result
d = abs(expect-actual);

% Result string
% result = sprintf('[%s] %s\n\tExpect = %d; Actual = %d; Diff = %d', caller, msg, expect, actual, d);
result = sprintf('%s: Expect = %d; Actual = %d; Diff = %d', msg, expect, actual, d);

if d < tol
    fprintf('%s < %d', result, tol)
    fprintf('\tPASS\n')
    pass = 1;
else
    fprintf('%s > %d', result, tol)
    fprintf('\t[\b---- FAIL ----]\b\n')
    pass = 0;
end

