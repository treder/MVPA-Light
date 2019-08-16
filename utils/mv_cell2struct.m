function s = mv_cell2struct(c)
% Utility function that converts a cell array of key-value pairs into a
% struct. For instance, the cell array {'n' 5 'kernel' 'rbf'} will be
% transformed into a struct s with 
%
%    s.n = 5
%    s.kernel = 'rbf'
%
% Usage:
% s = mv_cell2struct(c)
%
%Parameters:
% c          - cell array with key-value pairs {'key1' value1 'key2' value2...}
%
%Output:
% s - struct 

% (c) Matthias Treder 

if isstruct(c), s = c; return; end

s = struct();

for ii=1:2:numel(c)
    s.(c{ii}) = c{ii+1};
end