function mv_check_clabel(clabel)
% Checks whether the class labels are integers 1, 2, 3 ... 

clabel = clabel(:);

if ~all(ismember(1:max(clabel),clabel))
    error('The vector of labels must consist of integers 1 (class 1), 2 (class 2), 3 (class 3) and so on')
end
