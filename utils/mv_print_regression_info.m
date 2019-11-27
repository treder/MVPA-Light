function mv_print_regression_info(cfg, X, Y)
% Prints information regarding the regression procedure, including the
% cross-validation procedure and the dimensions of the input data. This
% function is called by mv_regress.
%
% Usage:
% mv_print_regression_info(cfg, <X, Y, X2, Y2>)
%
%Parameters:
% cfg       - configuration struct with parameters .model, .k, .cv,
%             and .repeat
% X         - [... x ... x ... x] dataset (optional)
% Y         - [samples x ...] responses  (optional)
% X2        - [... x ... x ... x] second dataset (optional)
% Y2        - [samples x ...] second responses dataset (optional)

if nargin <= 3
    % Print type of regression
    if ~strcmp(cfg.cv,'none')
        if strcmp(cfg.cv,'kfold'), k=sprintf(' (k=%d)', cfg.k); else k=''; end
        fprintf('Performing %s cross-validation%s with %d repetitions using a %s model.\n', ...
            cfg.cv, k, cfg.repeat, upper(cfg.model))
    else
        fprintf('Training and testing on the same data using a %s model.\n',upper(cfg.model))
    end
    
    % Print data information
    if nargin>1 
        if isempty(cfg.dimension_names)
            fprintf('Data is %s.\n', strjoin(cellfun(@num2str, num2cell(size(X)),'Un',0),' x ') )
        else
            fprintf('Data is %s.\n', strjoin(cellfun(@(x,y) [num2str(x) ' ' y], num2cell(size(X)), cfg.dimension_names,'Un',0),' x ') )
        end
    end
    
    % Print response information
    %%% TODO : do we want to print the range of the response values here?
    if nargin>2
%         u = unique(Y);
%         fprintf('Class frequencies: ');
%         for ii=1:numel(u)
%             fprintf('%2.2f%% [class %d]', 100*sum(Y==u(ii))/numel(Y), u(ii));
%             if ii<numel(u), fprintf(', '), end
%         end
%         fprintf('.\n')
    end

end
