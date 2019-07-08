function run_all_unittests()
% Call this function to execute all unit tests contained in this
% subfolder.

global FAIL_COUNT
FAIL_COUNT = 0;

path = fileparts(mfilename('fullpath'));

% Find all other unit tests in unittest subfolder
unittests = dir([path filesep 'unittest_*.m']);

% Execture every unit test
for uu=1:numel(unittests)
    fprintf('\n-------- Executing %s --------\n', unittests(uu).name)
    feval(unittests(uu).name(1:end-2))
end

% Print number of fails
if FAIL_COUNT > 0
    fprintf('\n[\bUnit testing failed (%d fails).]\b\n', FAIL_COUNT)
else
    fprintf('\n[\bUnit testing succeeded (no fails).]\b\n')
end
