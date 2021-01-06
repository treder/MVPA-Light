function pparam = mv_fix_preprocess_dimensions(pparam, dim_order)
% mv_classify and mv_regress rearrange the data dimensions such that the
% dimensions are in the order [samples, search dimensions, features].
% The preprocess parameters have some fields referring to dimensions; these
% need to be adapted according to the new dim_order.
%
% Usage:
%  pparam = mv_fix_preprocess_dimensions(pparam, dim_order)
%
%Parameters:
% pparam         - struct or cell array of preprocessing parameter structs
% dim_order      -
%
% cfg     - struct with preprocessing parameters:
% .preprocess         - cell array containing the preprocessing pipeline. The
%                       pipeline is applied in chronological order
% .preprocess_param   - cell array of preprocessing parameter structs for each
%                       function. Length of preprocess_param must match length
%                       of preprocess

if ~iscell(pparam)
    to_cell = 1;
    pparam = {pparam}; 
else
    to_cell = 0;
end

for pp=1:numel(pparam)   % -- loop over preprocessing pipeline
    
   % Look for dimension paramaters
   fn = fieldnames(pparam{pp});
   dim_ix = find(contains(fn,'dimension'));
   
   for ix = 1:numel(dim_ix)
       % Change dimension according to new dim_order
       param_name = fn{dim_ix(ix)};
       pparam{pp}.(param_name) = find(ismember(dim_order, pparam{pp}.(param_name)));
   end
   

end

if to_cell
    pparam = pparam{1};
end