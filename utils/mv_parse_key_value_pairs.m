function opt = mv_parse_key_value_pairs(varargin)
% Converts key-value pairs into an options structure.
%
% Usage:
% opt = mv_util_parse_key_value_pairs(...,key1,value1,key2,value2,...)
%
% Parameters:
% key1, key2, ...       - string denoting the name of the parameter
% value1, value2, ...   - value denoting the parameter value
%
% Output:
% opt   - options structure with the format opt.key1= value1 and so on

opt = struct();
vv=1;
while vv < numel(varargin)
    if ischar(varargin{vv})
        opt.(varargin{vv}) = varargin{vv+1};
        vv = vv+2;
    else
        vv = vv+1;
    end
end
