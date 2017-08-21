function varargout = mv_check_labels(label)
%Checks a vector of binary class labels. Returns the labels for classes 1
%and 2. As third argument, returns a vector where class 1 is denoted by 1's
%and class 2 is denoted by -1's.
%
% Returns:
% l1        - first label (1)
% l2        - second label
% labels    - fixed labels with 1's (class 1) and -1's (class 2)

% Check the labels
u=unique(label);
if numel(u)~= 2
    error('there are %d different label types, should be only 2',numel(u))
end
if any(u==1)
    l1= 1;
    l2= u(u~=1);
else
    l1= u(1);
    l2= u(2);
    warning('Class label 1 should be used to denote class 1, instead %d is used',l1)
end

varargout{1}= l1;
varargout{2}= l2;

if nargout > 2
    tmp = double(label);
    varargout{3}(tmp==l1) = 1;
    varargout{3}(tmp==l2) = -1;
end
